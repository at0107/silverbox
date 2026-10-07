| name | width | depth | act | opt | lr | steps | win | win_final | skip | phys_init | seed | n_params | train_rmse | test_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a02_tanh32x2_w64 | 32 | 2 | tanh | adam | 0.003 | 3000 | 64 |  | False | False | 0 | 1217 | 0.00173 | 0.00842 |
| a05_relu64x3_w64 | 64 | 3 | relu | adam | 0.003 | 3000 | 64 |  | False | False | 0 | 8641 | 0.00171 | 0.00327 |
| a11_tanh64x3_rmsprop_w64 | 64 | 3 | tanh | rmsprop | 0.001 | 3000 | 64 |  | False | False | 0 | 8641 | 0.00220 | 0.01473 |
| a01_tanh64x3_w64 | 64 | 3 | tanh | adam | 0.003 | 3000 | 64 |  | False | False | 0 | 8641 | 0.00176 | 0.00847 |
| a10_tanh64x3_lr1e-3_w64 | 64 | 3 | tanh | adam | 0.001 | 3000 | 64 |  | False | False | 0 | 8641 | 0.00179 | 0.00855 |
| a04_silu64x3_w64 | 64 | 3 | silu | adam | 0.003 | 3000 | 64 |  | False | False | 0 | 8641 | 0.00169 | 0.00216 |
| a06_tanh64x3_skip_w64 | 64 | 3 | tanh | adam | 0.003 | 3000 | 64 |  | True | False | 0 | 8644 | 0.00174 | 0.00593 |
| a07_tanh64x3_skipphys_w64 | 64 | 3 | tanh | adam | 0.003 | 3000 | 64 |  | True | True | 0 | 8644 | 0.00166 | 0.00247 |
| a03_tanh128x3_w64 | 128 | 3 | tanh | adam | 0.003 | 3000 | 64 |  | False | False | 0 | 33665 | 0.00166 | 0.00936 |
| a08_tanh64x3_w128 | 64 | 3 | tanh | adam | 0.003 | 3000 | 128 |  | False | False | 0 | 8641 | 0.00153 | 0.00865 |
| a09_tanh64x3_curr32-256 | 64 | 3 | tanh | adam | 0.003 | 4000 | 32 | 256 | False | False | 0 | 8641 | 0.00147 | 0.00758 |
| a12_tanh64x3_skipphys_curr32-256 | 64 | 3 | tanh | adam | 0.003 | 4000 | 32 | 256 | True | True | 0 | 8644 | 0.00147 | 0.00185 |
| s2_a07_seed2 | 64 | 3 | tanh | adam | 0.003 | 4000 | 64 |  | True | True | 2 | 8644 | 0.00161 | 0.00166 |
| s2_a04_seed1 | 64 | 3 | silu | adam | 0.003 | 4000 | 64 |  | False | False | 1 | 8641 | 0.00161 | 0.00236 |
| s2_a07_seed0 | 64 | 3 | tanh | adam | 0.003 | 4000 | 64 |  | True | True | 0 | 8644 | 0.00160 | 0.00180 |
| s2_a04_seed2 | 64 | 3 | silu | adam | 0.003 | 4000 | 64 |  | False | False | 2 | 8641 | 0.00161 | 0.00223 |
| s2_a07_seed1 | 64 | 3 | tanh | adam | 0.003 | 4000 | 64 |  | True | True | 1 | 8644 | 0.00160 | 0.00137 |
| s2_a04_seed0 | 64 | 3 | silu | adam | 0.003 | 4000 | 64 |  | False | False | 0 | 8641 | 0.00160 | 0.00223 |
| s2_a12_seed0 | 64 | 3 | tanh | adam | 0.003 | 4000 | 32 | 256 | True | True | 0 | 8644 | 0.00147 | 0.00185 |
| s2_a12_seed2 | 64 | 3 | tanh | adam | 0.003 | 4000 | 32 | 256 | True | True | 2 | 8644 | 0.00147 | 0.00148 |
| s2_a12_seed1 | 64 | 3 | tanh | adam | 0.003 | 4000 | 32 | 256 | True | True | 1 | 8644 | 0.00145 | 0.00139 |
