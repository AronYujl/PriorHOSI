"""Independent-development diagnostics for one globally fixed evidence scale."""
import json
from pathlib import Path

import numpy as np
import torch


def save_development_episode(editor, record_start, output_dir, episode, windows, seconds):
    records = editor.records[record_start:]
    stem = 'episode-%03d' % episode['canonical_ordinal']
    payload = dict(episode=episode, window_count=len(windows), records=records,
                   generation_seconds=seconds, quality_evaluated=False,
                   metric_scope='relation/voxel window proxies; no native mesh-SDF evaluation')
    directory = Path(output_dir)
    with (directory / (stem + '.json')).open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    with (directory / (stem + '.pt')).open('xb') as handle:
        torch.save(dict(episode=episode, windows=windows, corrections=editor.motion_records), handle)
    editor.motion_records.clear()
    return payload


def estimate_global_scale(episodes, calibration_scenes, target_ratio=.1,
                          minimum_active_windows=6):
    """Scene-balanced median of positive-scene source gradient norm ratios.

    Inputs must be passive lambda1 records. No verification row participates in
    selection, and no tiny-signal denominator is replaced by an arbitrary floor.
    """
    by_scene = {scene: [] for scene in calibration_scenes}
    inactive = zero_signal = 0
    for episode in episodes:
        scene = episode['episode']['scene_name']
        if scene not in by_scene:
            continue
        for record in episode['records']:
            if record['mode'] != 'calibrate':
                raise ValueError('scale estimation requires passive source records')
            scene_energy = record['source_terms']['human_scene'][0] + record['source_terms']['object_scene'][0]
            if scene_energy == 0:
                inactive += 1
                continue
            ratios = []
            for iteration in record['iterations']:
                gradients = iteration['parameter_gradient_norms']
                numerator, denominator = gradients['explicit'][0], gradients['hsi_evidence'][0]
                if not np.isfinite(numerator) or not np.isfinite(denominator):
                    raise ValueError('nonfinite source gradient')
                if denominator == 0:
                    zero_signal += 1
                else:
                    ratios.append(numerator / denominator)
            if ratios:
                by_scene[scene].append(float(np.median(ratios)))
    if any(not values for values in by_scene.values()) or sum(map(len, by_scene.values())) < minimum_active_windows:
        raise ValueError('insufficient positive-scene calibration coverage')
    scene_medians = {scene: float(np.median(values)) for scene, values in by_scene.items()}
    unrounded = target_ratio * float(np.median(list(scene_medians.values())))
    coefficient = float(format(unrounded, '.2g'))
    return dict(lambda_dp=coefficient, unrounded=unrounded, target_ratio=target_ratio,
                calibration_scenes=list(calibration_scenes), scene_medians=scene_medians,
                active_windows={scene: len(values) for scene, values in by_scene.items()},
                inactive_windows=inactive, zero_hsi_level_count=zero_signal,
                rule='target times median(scene median(window median(level norm ratios))), two significant figures')
