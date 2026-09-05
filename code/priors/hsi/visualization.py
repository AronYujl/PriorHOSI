"""Prepare synchronized generated/GT mesh caches for qualitative review."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from utils import SMPLX_JOINTS_28, create_smplx_model, run_smplx_model


def paired_frame_indices(frame_count: int, stride: int):
    """One common clock for both arms; playback rate is reduced by stride."""
    return np.arange(0, frame_count, stride, dtype=np.int64)


def joint_comparison(generated: torch.Tensor, truth: torch.Tensor):
    distance = (generated - truth).norm(dim=-1)
    return {
        "joint_error_cm": float(distance.mean().item() * 100),
        "root_error_cm": float(distance[:, 0].mean().item() * 100),
        "final_root_error_cm": float(distance[-1, 0].item() * 100),
        "root_height_error_cm": float(
            (generated[:, 0, 1] - truth[:, 0, 1]).abs().mean().item() * 100
        ),
        "per_frame_joint_error_cm": (distance.mean(dim=-1) * 100).cpu().tolist(),
    }


def _motion_paths(root):
    paths = {}
    for path in sorted(Path(root).rglob("motion/*.npz")):
        with np.load(path) as data:
            paths[str(data["sequence_id"])] = path
    return paths


def _metric_records(root):
    records = {}
    for path in sorted(Path(root).rglob("evaluation/per_sequence_metrics.json")):
        records.update(json.loads(path.read_text())["metrics"])
    return records


def _reconstruct(data, model, device):
    local = np.concatenate(
        [data["global_orient"][:, None], data["body_pose"]], axis=1
    )
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, len(local), 128):
            pose = torch.as_tensor(local[start:start + 128], device=device)
            transl = torch.as_tensor(data["transl"][start:start + 128], device=device)
            betas = torch.as_tensor(data["betas"], device=device)[None].expand(len(pose), -1)
            v, j = run_smplx_model(
                pose, transl, betas, str(data["gender"]),
                joints_ind=SMPLX_JOINTS_28, smpl_model=model,
            )
            vertices.append(v)
            joints.append(j)
    return torch.cat(vertices), torch.cat(joints)


def _write_index(root: Path, rows):
    cards = []
    for row in rows:
        case = html.escape(row["case_id"])
        caption = html.escape(row["caption"])
        metrics = row["comparison"]
        cards.append(
            '<article id="%s" data-action="%s"><h2>%s · %s</h2><p>%s · %s · %.2f s</p>'
            '<video controls preload="metadata" poster="%s/poster.png" src="%s/comparison.mp4"></video>'
            '<p>Recorded-span joint difference %.1f cm · whole-export root difference %.1f cm · final root difference %.1f cm</p>'
            '<p>Recorded GT duration %.2f s; later exported frames hold the GT endpoint. Orange video header marks this interval.</p>'
            '<p><a href="%s/keyframes.png">Matched keyframes</a> · '
            '<a href="%s/teaser_pair.png">Overlaid poses</a> · '
            '<a href="%s/comparison.json">Measurements and provenance</a></p></article>'
            % (case, row['action_class'], case, caption, row["scene_name"], row["sequence_id"], row["duration_seconds"],
               case, case, float(np.mean(metrics['per_frame_joint_error_cm'][:row['source_length']])), metrics["root_error_cm"],
               metrics["final_root_error_cm"], row['source_length']/30., case, case, case)
        )
    doc = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>HSIPrior: generated and ground truth</title><style>'
           'body{max-width:1320px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#22303a;background:#f5f6f7}'
           'h1{font-size:30px}h2{font-size:21px}article{background:white;padding:24px;margin:24px 0;border-radius:12px}'
           'video{width:100%;background:white}a{color:#186c88}p{margin:10px 0}'
           'nav{position:sticky;top:0;background:#f5f6f7;padding:12px 0;z-index:2}'
           'button{padding:9px 18px;border:1px solid #ccd5d9;border-radius:7px;background:white;margin:0 8px 6px 0;cursor:pointer}</style>'
           '<h1>HSIPrior · 20 paired training-scene candidates</h1>'
           '<p><strong>Left: ground truth. Right: R2 + CG generation.</strong> Same source history, body, room, camera and clock.</p>'
           '<p>R2 final EMA · 500 diffusion steps · posterior-coefficient guidance · seed 42. '
           'The first two coarse frames are supplied; subsequent windows use generated history. '
           'Source heading is preserved. GT is resampled with the same stride-3/interpolation protocol. '
           'Furniture and human trajectories are unchanged. Root/joint differences are frame-aligned comparisons to one recording, '
           'not measures of the only valid motion. All 20 candidates, including unsuccessful actions, are retained.</p>'
           '<p><a href="selection.json">Source selection and measurements</a></p>'
           '<nav><button onclick="showCases(\'all\')">全部20例</button>'
           '<button onclick="showCases(\'navigate\')">行走</button><button onclick="showCases(\'sit\')">坐下</button>'
           '<button onclick="showCases(\'rise\')">起身</button><button onclick="showCases(\'lie\')">躺卧</button>'
           '<button onclick="showCases(\'wash\')">洗手</button></nav>'
           + ''.join(cards) + '<script>function showCases(action){document.querySelectorAll("article").forEach(card=>{'
           'card.style.display=(action==="all"||card.dataset.action===action)?"":"none";});}</script></html>')
    (root / 'index.html').write_text(doc, encoding='utf-8')


def prepare_paired_review(cfg):
    """Rebuild both exported FK paths on CUDA and save presentation-only caches."""
    selection = json.loads(Path(cfg.qualitative_selection).read_text())
    generated = _motion_paths(cfg.qualitative_generated_dir)
    truth = _motion_paths(cfg.qualitative_gt_dir)
    generated_metrics = _metric_records(cfg.qualitative_generated_dir)
    truth_metrics = _metric_records(cfg.qualitative_gt_dir)
    root = Path(cfg.qualitative_review_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(cfg.device))
    stride = int(cfg.qualitative_render_stride)
    models, rows = {}, []
    selected_ids = cfg.get("qualitative_case_ids", None)
    for source in selection['episodes']:
        if selected_ids is not None and source['case_id'] not in selected_ids:
            continue
        sequence = source['sequence_id']
        case_dir = root / source['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        with np.load(generated[sequence]) as data:
            pred = dict(data)
        with np.load(truth[sequence]) as data:
            gt = dict(data)
        gender = str(gt['gender'])
        if gender not in models:
            models[gender] = create_smplx_model(gender, device)
        vertices, joints = [], []
        for motion in (gt, pred):
            v, j = _reconstruct(motion, models[gender], device)
            vertices.append(v)
            joints.append(j)
        torch.cuda.synchronize(device)
        stats = joint_comparison(joints[1], joints[0])
        stats['initial_coarse_joint_max_error_cm'] = float(
            np.linalg.norm(pred['global_jpos'][:2] - gt['global_jpos'][:2], axis=-1).max() * 100
        )
        indices = paired_frame_indices(len(vertices[0]), stride)
        cache = torch.stack([v[indices][..., [0, 2, 1]] for v in vertices])
        cache[..., 1] *= -1
        np.save(case_dir / 'paired_vertices.npy', cache.cpu().numpy())
        np.save(case_dir / 'human_faces.npy', np.asarray(models[gender].faces, dtype=np.int32))
        np.save(case_dir / 'frame_indices.npy', indices)
        cache_bounds = torch.stack([cache.amin(dim=(0, 1, 2)), cache.amax(dim=(0, 1, 2))]).cpu().tolist()
        root_rotation = axis_angle_to_matrix(torch.as_tensor(gt['global_orient'][0], dtype=torch.float32))
        forward = root_rotation @ torch.tensor([0., 0., 1.])
        azimuth = float(np.degrees(np.arctan2(-float(forward[2]), float(forward[0]))) - 85)
        config = {
            'case_id': source['case_id'], 'caption': source['caption'],
            'vertices': str(case_dir / 'paired_vertices.npy'),
            'faces': str(case_dir / 'human_faces.npy'),
            'scene_mesh': str(Path(cfg.lingo_mesh_root) / source['scene_name'] / 'mesh_low.obj'),
            'output_dir': str(case_dir), 'fps': float(cfg.fps) / stride,
            'frame_count': len(indices), 'source_frame_indices': indices.tolist(),
            'keyframes': np.linspace(0, len(indices) - 1, 5).round().astype(int).tolist(),
            'camera_azimuth_degrees': azimuth, 'camera_elevation_degrees': 38.0,
            'motion_bounds': cache_bounds, 'width': 640, 'height': 512,
            'samples': 12, 'engine': 'CYCLES', 'video': True,
            'figure_width': 1400, 'figure_height': 1120,
        }
        (case_dir / 'render_config.json').write_text(json.dumps(config, indent=2) + '\n')
        row = dict(source)
        row.update({
            'generated_motion': str(generated[sequence]), 'ground_truth_motion': str(truth[sequence]),
            'comparison': stats, 'generated_metrics': generated_metrics[sequence],
            'ground_truth_metrics': truth_metrics[sequence],
            'duration_seconds': len(indices) / config['fps'],
            'render_fps': config['fps'], 'source_heading_preserved': True,
            'ground_truth_resampling': 'stride3; same interpolation and terminal padding as generation',
            'body_gender': gender, 'betas_equal': bool(np.array_equal(gt['betas'], pred['betas'])),
            'render_config': str(case_dir / 'render_config.json'),
        })
        (case_dir / 'comparison.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        print('PAIRED_CACHE %s frames=%d initial_max=%.7fcm joint=%.2fcm' % (
            source['case_id'], len(indices), stats['initial_coarse_joint_max_error_cm'], stats['joint_error_cm']), flush=True)
    report = dict(selection)
    report['episodes'] = rows
    (root / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    _write_index(root, rows)
    return root / 'selection.json'


def finalize_paired_review(cfg):
    """Encode synchronized pairs and lay out exact common-frame comparisons."""
    from PIL import Image, ImageDraw, ImageFont

    root = Path(cfg.qualitative_review_dir)
    selection = json.loads((root / 'selection.json').read_text())
    selected_ids = cfg.get('qualitative_case_ids', None)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22)
    small_font = ImageFont.truetype(font_path, 16)
    for row in selection['episodes']:
        if selected_ids is not None and row['case_id'] not in selected_ids:
            continue
        folder = root / row['case_id']
        config = json.loads((folder / 'render_config.json').read_text())
        width, height = config['width'], config['height']
        video_filter = (
            'hstack=inputs=2,pad=iw:ih+60:0:60:color=white,'
            "drawbox=x=0:y=39:w=%d:h=21:color=0xffddaa:t=fill:enable='gte(t,%.6f)',"
            'drawtext=fontfile=%s:text=Ground truth:x=20:y=12:fontsize=23:fontcolor=0x243440,'
            'drawtext=fontfile=%s:text=R2 + CG:x=%d:y=12:fontsize=23:fontcolor=0x243440,'
            "drawtext=fontfile=%s:text='Recorded GT %.2f s; orange indicates endpoint hold':x=20:y=42:fontsize=13:fontcolor=0x243440"
        ) % (width, row['source_length']/30., font_path, font_path, width+20,
             font_path, row['source_length']/30.)
        video = folder / 'comparison.mp4'
        if not video.exists():
            subprocess.run([
                str(cfg.qualitative_ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
                '-framerate', str(config['fps']), '-i', str(folder/'ground_truth'/'%05d.png'),
                '-framerate', str(config['fps']), '-i', str(folder/'generated'/'%05d.png'),
                '-filter_complex', video_filter, '-c:v', 'h264_nvenc', '-preset', 'p4',
                '-cq', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video),
            ], check=True)
        poster = Image.new('RGB', (width*2, height+60), 'white')
        frame = config['keyframes'][3]
        for arm, name in enumerate(('ground_truth', 'generated')):
            poster.paste(Image.open(folder/name/('%05d.png' % frame)), (arm*width, 60))
        draw = ImageDraw.Draw(poster)
        draw.text((20, 12), 'Ground truth', fill='#243440', font=font)
        draw.text((width+20, 12), 'R2 + CG', fill='#243440', font=font)
        poster.save(folder/'poster.png')
        cell_width, cell_height = 400, round(height*400/width)
        sheet = Image.new('RGB', (cell_width*5, (cell_height+58)*2+70), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), '%s | %s | identical camera and timestamps' % (
            row['case_id'], row['caption']), fill='#243440', font=font)
        for arm, name in enumerate(('ground_truth', 'generated')):
            top = 70 + arm*(cell_height+58)
            draw.text((16, top), 'Ground truth' if arm==0 else 'R2 + CG', fill='#243440', font=font)
            for column, frame in enumerate(config['keyframes']):
                image = Image.open(folder/name/('%05d.png' % frame)).resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                sheet.paste(image, (column*cell_width, top+32))
                time = config['source_frame_indices'][frame]/30.
                label = 't = %.2f s' % time
                if arm == 0 and time >= row['source_length']/30.:
                    label += ' | beyond recorded duration'
                draw.text((column*cell_width+16, top+34+cell_height), label,
                          fill='#243440', font=small_font)
        sheet.save(folder/'keyframes.png')
        gt_image = Image.open(folder/'teaser_ground_truth.png')
        pred_image = Image.open(folder/'teaser_generated.png')
        pair = Image.new('RGB', (gt_image.width*2, gt_image.height+70), 'white')
        pair.paste(gt_image, (0, 70))
        pair.paste(pred_image, (gt_image.width, 70))
        draw = ImageDraw.Draw(pair)
        draw.text((25, 20), 'Ground truth', fill='#243440', font=font)
        draw.text((gt_image.width+25, 20), 'R2 + CG', fill='#243440', font=font)
        pair.save(folder/'teaser_pair.png')
        print('PAIRED_MEDIA', row['case_id'], flush=True)
    return root/'index.html'
