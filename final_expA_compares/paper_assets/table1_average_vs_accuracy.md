| Method | Accuracy_best | Accuracy_last50 | avg_B_c | avg_A_c_ratio | avg_G_drift | Source_JSON |
|---|---|---|---|---|---|---|
| SFL IID | 0.5990 | 0.5949 | 0.9916 | 0.7927 | 0.0054 | final_expA_compares/result-sfl-resnet18_flex-cifar10-dist-uniform-sel-uniform-agg-fedavg-split--bs-50-20260210-190559.json |
| USFL non-IID | 0.5335 | 0.5130 | 1.0008 | 0.8189 | 0.0120 | final_expA_compares/result-usfl-resnet18_flex-cifar10-dist-shard_dirichlet-alpha-0.3-sel-usfl-agg-fedavg-split--bs-500-gradshuf-gradshuf-random-dbs-20260210-181701.json |
| GAS non-IID | 0.3757 | 0.3336 | 0.9761 | 0.5501 | 0.0017 | final_expA_compares/results_gas_cifar10_20260212_184417.json |
| MultiSFL non-IID | 0.3665 | 0.2707 | 0.9843 | 0.7176 | 0.0217 | final_expA_compares/results_multisfl_cifar10_shard_dirichlet_20260212_222525.json |
| MIX2SFL non-IID | 0.3617 | 0.2493 | 0.9115 | 0.6688 | 0.0019 | final_expA_compares/result-mix2sfl-resnet18_flex-cifar10-dist-shard_dirichlet-alpha-0.3-sel-uniform-agg-fedavg-split--bs-50-20260210-010449.json |
| SFL non-IID | 0.3239 | 0.1685 | 1.0231 | 0.7912 | 0.0108 | final_expA_compares/result-sfl-resnet18_flex-cifar10-dist-shard_dirichlet-alpha-0.3-sel-uniform-agg-fedavg-split--bs-50-20260210-181700.json |
