# TinyPAD: code for "A Multimodal TinyML and Edge-AI Framework for Continuous Health Anomaly Detection Using Wearable IoT Sensor Networks"

All experiments were run on a RunPod pod (NVIDIA RTX A4000, runpod/pytorch:2.4.0 image).

## Data (public)
- WESAD: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx (unzip to `../raw/WESAD`)
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/ (to `../raw/mitdb`)

## Pipeline (`run_all.sh`)
1. `prepare_health.py ../raw ../data all`: builds `wesad.npz` (30-s windows, 5-s stride, 7 modalities) and `mitbih.npz` (beats, 2 leads + RR vector, DS1/DS2), plus the meta JSON files.
2. `dev_select.py` / `dev_select2.py`: design selection on validation people only (personalisation; fusion and modality-dropout rate).
3. `run_health.py --exp main|ablation|latency`: 7 methods x (WESAD LOSO, 3 seeds; MIT-BIH DS1->DS2, 5 seeds), ablations (seeds 0-2), latency, missing-sensor, calibration-length, int8 and cascade evaluations.
4. `analyze_health.py --results <results> --data ../data --out ../paper_assets`: summary.json, statistics, Figs 2-9.
5. `fig1_tinypad.py`: architecture figure.

`healthlib.py` contains the data loader, the personal normaliser, TinyPAD and the baselines, training and metrics.

## Provenance notes
- The WESAD TinyPAD runs (seeds 0-2) were re-run after a fix to the sensor on-time denominator in `run_health.py::sensor_gating` (`code_sha256_rerun.txt`; run_health.py sha256 51065cad...). The superseded files are kept in `runpod_results/results/superseded_gating_denominator/` and are not reported. GPU training is not bit-wise deterministic, so the re-run's WESAD TinyPAD scores differ slightly from the superseded files.
- `code_sha256.txt` lists the hashes of the code used for the remaining runs; `healthlib.py` and `prepare_health.py` are identical in both.
