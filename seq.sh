#!/bin/bash

python scripts/compute_score.py --script ttd --analysis model_srpz_full --model plssvd --dataset things_eeg_2 --uid openclip_rn50_yfcc15m  > "artifacts/output_0.txt" 2>&1
python scripts/compute_score.py --script ttd --analysis model_srpz_full --model plssvd --dataset things_meg --uid openclip_rn50_yfcc15m  > "artifacts/output_1.txt" 2>&1