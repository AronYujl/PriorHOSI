# R4 GT body FK loss calibration

Seed42, 4×512 first real training batch, random initialization, zero updates. Selected body weight: 0.248127122918.

| Term | Weight | Median raw loss | Weighted loss | Trunk gradient norm | Rotation-head gradient norm |
|---|---:|---:|---:|---:|---:|
| jpos | 1 | 0.4453515 | 0.4453515 | 1.560349 | 0 |
| jrot | 1 | 0.6355843 | 0.6355843 | 0.8681446 | 0.8978299 |
| hand_foot_fk | 3 | 7.301933 | 21.9058 | 176.4544 | 14.29379 |
| fullbody_seam | 0.54945 | 4.847594 | 2.66351 | 96.31511 | 4.917819 |
| body_fk | 0.2481271 | 0.1342056 | 0.03330006 | 0.8568844 | 0.9046068 |
| r2_total | 1 | 25.65099 | 25.65099 | 581.2633 | 45.03852 |

Norm columns describe unweighted terms. The new term is 25% of rotation L1 on the rotation head and about 0.0366% of the R2 total on the trunk at this initial batch. The old geometry terms dominate that trunk total; its 10% rule alone would yield 67.8345. Calibration measures initial gradient scale, not converged efficacy.

Metric mapping: jpos → direct-position FID/semantic features; jrot → orientation; hand/foot FK → endpoint geometry; fullbody seam → cross-window acceleration; new body FK → GT body shape and pose across the full future. Physical quality, contact engagement, FID and semantics remain rollout acceptance questions.

Per-rank values and cosines are in the adjacent JSON. Existing input manifests are referenced by the calibration manifest. The zero-update workload exited 0; CUDA shutdown warnings occurred while persistent data workers closed.
