# code/instacart_train — the Instacart matched pair and the MLP

Regenerates the second dataset's prediction arrays from the public Instacart 2017 data (six CSV files,
Kaggle competition *Instacart Market Basket Analysis*; SHA-256 of the files in `data/instacart/SHA256SUMS_raw.txt`).

Formulation (next-basket top-K, not the Kaggle reorder task): one row per user and reference order;
features from the user's prior orders only; within-user chronological split — train on order n−2,
calibrate on order n−1 (validation), test on order n (the user's last order); labels = the products of
the reference order over the top-N products (N chosen as "at least 20 training positives, cut by
compute budget", recorded in `label_meta.json`). MAP@7 follows the Kaggle definition (rows with at
least one positive).

```bash
# 1. put the six CSV files (or the Kaggle zip) under data/instacart/raw/
python instacart_io.py --help                     # cache of the CSVs (read once)
python build_features.py --raw-dir ../../data/instacart/raw --out-dir ../../data/instacart/data/processed --max-labels 4000 --self-check
python eda.py --processed-dir ../../data/instacart/data/processed --out ../../data/instacart/results/eda.json
# 2. the matched LightGBM pair: same configuration with and without per-label weights, plus leaf-step caps
python train_matched_pair.py --processed-dir ../../data/instacart/data/processed --output-dir ../../data/instacart/outputs_matched_pair \
    --configs unweighted,weighted,weighted_mds0.3,weighted_mds0.7,weighted_mds1,weighted_mds2,weighted_mds5 --n-jobs 8 --num-threads 1
python train_matched_pair.py ... --configs weighted_sub0,weighted_sub1,weighted_sub2,unweighted_sub0,unweighted_sub1,unweighted_sub2   # 90% train-row draws
# 3. the shared-trunk MLP with per-label pos_weight (all products; w in {1, n-/n+})
python train_mlp_posweight.py --dataset instacart --processed-dir ../../data/instacart/data/processed --output-dir ../../data/instacart/outputs_mlp
python train_mlp_posweight.py --dataset santander --features-cache ../../data/santander/outputs_matched_pair/features_cache --output-dir ../../data/santander/outputs_mlp
```

Every run writes `predictions.npz` (`p_te`, `p_va`, `Y_te`, `Y_va`) and `run_meta.json` (counts,
hyper-parameters, per-label positives and weights; the shipped copies are under `data/instacart/`).
The rederive code (`../run_experiment.py --dataset instacart|instacart_mlp`) reads these arrays and
recomputes every Instacart number of the paper.
