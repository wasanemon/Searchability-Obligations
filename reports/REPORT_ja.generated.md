# 実験生データからの自動集計（最終報告データ節素材）

生成時刻 (UTC): `2026-09-05T12:35:42.125778+00:00`

この文書は保存済み JSONL と manifest のみから再生成した集計素材であり、研究仮説の最終判断を自動で行わない。`final` 指定の未完了 run は集計から fail-closed で除外し、下表に状態を残す。その他の未完了 run は `完了` 列で区別する。

## 入力と完全性

- run 数: 2（完了 marker あり: 2）
- 未完了のため除外した final run: 1
- query-method 生行数: 112800
- 契約違反として保存された行数: 0
- optimized baseline 検証失敗行数: 0
- 独立 Fraction oracle: 実行 16 / mismatch 0 / 上限による skip 0
- raw shard は checkpoint 記載の SHA-256 と行数を再検証済み。検索処理は分析時に再実行していない。

### Run 識別子

| run ID | 完了 | config hash | dataset hash / split ID | 状態 |
|---|---:|---|---|---|
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` | yes | `98bb0214e0869353364c1b5560616c9ccb065b5b1f0c65e7243ab9eace1a3909` | gist-initial-group-build-seed-0: 958b61aea26a / 5775a602afce; gist-initial-group-build-seed-1: 157fa42874d5 / 67e2c70cae84; gist-initial-group-build-seed-2: 157fa42874d5 / 67e2c70cae84; sift-base-ef32: b6203d1ca68d / c7909e8e2d59; sift-base-ef512: b6203d1ca68d / c7909e8e2d59; sift-delta-0: cb8b2801163e / 7134dc58f45f; sift-delta-1000: 708fe33fa172 / da8282c05f07; sift-groups-16: b6203d1ca68d / c7909e8e2d59; sift-groups-512: b6203d1ca68d / c7909e8e2d59; sift-groups-64: b6203d1ca68d / c7909e8e2d59; sift-initial-group-build-seed-0: 82ac030797a4 / 455bb2ef951e; sift-initial-group-build-seed-1: b6203d1ca68d / c7909e8e2d59; sift-initial-group-build-seed-2: b6203d1ca68d / c7909e8e2d59; sift-k-1: b6203d1ca68d / c7909e8e2d59; sift-k-100: b6203d1ca68d / c7909e8e2d59 | completed |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` | yes | `e46edaa0851a1719b6894bc5c6783db31e5e64996b50e130391bc7149d8cf353` | sift-delta-100000: 6b408e07b3c1 / d4de9adf679f | completed |

### 集計から除外した未完了 final run

| run ID | 状態 | 完了 block / 全 block | failure | 除外理由 |
|---|---|---:|---|---|
| `texmex-main-and-lean-sweeps-20260905T062813.056342Z-98bb0214e0` | failed_incomplete | 104/176 | [{"completed_blocks": 104, "message": "memoryview: cannot cast view with zeros in shape or strides", "type": "TypeError"}] | `final_evidence_missing_COMPLETED_json` |

## 方式別集計

| run / 条件 | 方式 | 完了 | query | E2E p50/p95/p99 ms | 実測batch QPS p50 | 単query容量推定QPS | exact recall@k 平均 | skip率 | fallback率 | 違反 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `certified_full_delta_reference` | yes | 800 | N/A/N/A/N/A | N/A | N/A | 0.9879 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `delta_flat_full_scan` | yes | 800 | 111.3/113.3/114.7 | 9.103 | 8.956 | 0.9879 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `delta_hnsw_ef128` | yes | 800 | 110.3/111.9/113.3 | 9.193 | 9.043 | 0.9879 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `delta_hnsw_ef32` | yes | 800 | 110/111.5/113.2 | 9.274 | 9.067 | 0.9872 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `delta_hnsw_ef512` | yes | 800 | 110.9/112.5/113.8 | 9.207 | 8.994 | 0.9879 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `full_base_delta_exact_flat` | yes | 800 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `full_base_delta_flat_scan_performance` | yes | 800 | 116.1/117.5/119.5 | 8.509 | 8.591 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `full_base_delta_hnsw_ef128` | yes | 800 | 1.132/1.428/1.608 | 1058 | 900.8 | 0.9854 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `full_base_delta_hnsw_ef32` | yes | 800 | 0.5183/0.6437/0.6714 | 2453 | 1970 | 0.8847 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `full_base_delta_hnsw_ef512` | yes | 800 | 2.861/3.426/4.146 | 264.1 | 362.1 | 0.998 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `group_pruning_beta0` | yes | 800 | 176.9/179.9/183 | 5.635 | 5.648 | 0.9879 | 0.01945 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 800 | 211.6/215.1/220.3 | 4.672 | 4.727 | 0.9879 | 0.01945 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `group_pruning_beta_factor_0p01` | yes | 800 | 177.3/180.1/181.9 | 5.651 | 5.639 | 0.9879 | 0.01994 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `group_pruning_beta_factor_0p05` | yes | 800 | 177.1/180/181.9 | 5.663 | 5.645 | 0.9879 | 0.02262 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `group_pruning_beta_factor_0p1` | yes | 800 | 177/180/181.6 | 5.666 | 5.649 | 0.9879 | 0.02612 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-0` | `grouping_no_pruning_ablation` | yes | 800 | 177.3/180.3/182.9 | 5.632 | 5.631 | 0.9879 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `delta_flat_full_scan` | yes | 200 | 112.7/115.1/116 | 9.024 | 8.848 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `delta_hnsw_ef128` | yes | 200 | 111.7/114.1/115.2 | 9.105 | 8.923 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `delta_hnsw_ef512` | yes | 200 | 112.3/114.9/115.9 | 8.949 | 8.878 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `full_base_delta_flat_scan_performance` | yes | 200 | 116.8/119/160.4 | 8.517 | 8.45 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `full_base_delta_hnsw_ef128` | yes | 200 | 1.234/2.541/3.662 | 579.2 | 742.2 | 0.985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `full_base_delta_hnsw_ef512` | yes | 200 | 3.401/6.321/7.82 | 338.5 | 273.8 | 0.9975 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `group_pruning_beta0` | yes | 200 | 181.5/184.2/186.4 | 5.582 | 5.511 | 0.989 | 0.01539 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 215.1/218/221.2 | 4.666 | 4.661 | 0.989 | 0.01539 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `group_pruning_beta_factor_0p05` | yes | 200 | 181.6/184.9/201.5 | 5.552 | 5.496 | 0.989 | 0.01855 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-1` | `grouping_no_pruning_ablation` | yes | 200 | 181.3/184.7/188.9 | 5.573 | 5.505 | 0.989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `delta_flat_full_scan` | yes | 200 | 111.6/114.1/116.1 | 9.08 | 8.93 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `delta_hnsw_ef128` | yes | 200 | 110.7/112.9/115.6 | 9.163 | 9.006 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `delta_hnsw_ef512` | yes | 200 | 111.5/113.8/116.4 | 9.119 | 8.949 | 0.989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `full_base_delta_flat_scan_performance` | yes | 200 | 116.4/118.9/145.1 | 8.45 | 8.513 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `full_base_delta_hnsw_ef128` | yes | 200 | 1.216/2.421/4.252 | 922.1 | 753.7 | 0.985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `full_base_delta_hnsw_ef512` | yes | 200 | 3.136/5.802/10.62 | 271.3 | 291.3 | 0.9975 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `group_pruning_beta0` | yes | 200 | 178.8/181.9/184.4 | 5.638 | 5.591 | 0.989 | 0.01706 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 215.3/219.3/222.6 | 4.605 | 4.637 | 0.989 | 0.01706 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `group_pruning_beta_factor_0p05` | yes | 200 | 178.6/182.6/197.3 | 5.593 | 5.579 | 0.989 | 0.01988 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `gist-initial-group-build-seed-2` | `grouping_no_pruning_ablation` | yes | 200 | 179.1/182.5/185.4 | 5.619 | 5.575 | 0.989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.971 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `delta_flat_full_scan` | yes | 200 | 25.1/25.39/25.79 | 44.69 | 39.87 | 0.971 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `delta_hnsw_ef128` | yes | 200 | 24.88/25.19/25.86 | 45.33 | 40.22 | 0.971 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `delta_hnsw_ef512` | yes | 200 | 25.22/25.51/25.95 | 44.61 | 39.67 | 0.971 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.43/16.74/17.1 | 63.11 | 60.57 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4907/0.6162/1.102 | 2691 | 1975 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.227/1.501/2.059 | 1069 | 801 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `group_pruning_beta0` | yes | 200 | 89.53/99.44/102.2 | 11.5 | 11.17 | 0.971 | 0.1784 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 96.07/107.3/108.9 | 10.5 | 10.53 | 0.971 | 0.1784 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `group_pruning_beta_factor_0p05` | yes | 200 | 88.47/99.37/101 | 11.71 | 11.32 | 0.971 | 0.208 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef32` | `grouping_no_pruning_ablation` | yes | 200 | 98.18/101.8/102.4 | 10.08 | 10.16 | 0.971 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `delta_flat_full_scan` | yes | 200 | 25.32/25.6/26.25 | 43.42 | 39.52 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `delta_hnsw_ef128` | yes | 200 | 25.05/25.42/26.08 | 43.94 | 39.9 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `delta_hnsw_ef512` | yes | 200 | 25.43/25.79/26.68 | 43.18 | 39.33 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.46/16.69/17.83 | 61.5 | 60.54 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.468/0.564/0.8302 | 1279 | 2094 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.155/1.364/2.219 | 1119 | 843 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `group_pruning_beta0` | yes | 200 | 89.76/98.92/100.7 | 11.2 | 11.26 | 1 | 0.1791 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 96.16/106.9/108.1 | 10.25 | 10.53 | 1 | 0.1791 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `group_pruning_beta_factor_0p05` | yes | 200 | 88.2/98.1/99.63 | 11.39 | 11.41 | 1 | 0.2089 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-base-ef512` | `grouping_no_pruning_ablation` | yes | 200 | 97.87/101.4/102.5 | 10.06 | 10.19 | 1 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `delta_flat_full_scan` | yes | 200 | 24.24/24.5/24.59 | 46.48 | 41.23 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `delta_hnsw_ef128` | yes | 200 | 24.24/24.5/24.59 | 46.42 | 41.22 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `delta_hnsw_ef512` | yes | 200 | 24.25/24.46/24.58 | 46.43 | 41.23 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.03/16.24/16.29 | 61.96 | 62.39 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4272/0.5127/0.5363 | 1354 | 2343 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.076/1.264/1.338 | 1109 | 935.2 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `group_pruning_beta0` | yes | 200 | 24.95/25.22/25.37 | 44.1 | 40.08 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 25.08/25.34/25.48 | 43.19 | 39.86 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `group_pruning_beta_factor_0p05` | yes | 200 | 24.95/25.23/25.32 | 44.1 | 40.06 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-0` | `grouping_no_pruning_ablation` | yes | 200 | 24.95/25.26/25.36 | 43.79 | 40.04 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `delta_flat_full_scan` | yes | 200 | 24.09/24.35/24.63 | 46.15 | 41.52 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `delta_hnsw_ef128` | yes | 200 | 24.11/24.37/24.51 | 45.96 | 41.47 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `delta_hnsw_ef512` | yes | 200 | 24.26/24.51/24.79 | 45.82 | 41.21 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.36/16.58/17.56 | 63.69 | 60.79 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4227/0.5533/1.649 | 2593 | 2179 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.133/1.672/2.943 | 1087 | 827.2 | 0.9995 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `group_pruning_beta0` | yes | 200 | 49.04/52.75/53.01 | 21.16 | 20.37 | 0.9985 | 0.3145 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 49.72/53.31/53.87 | 20.85 | 20.13 | 0.9985 | 0.3145 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `group_pruning_beta_factor_0p05` | yes | 200 | 48.52/52.42/53.7 | 21.39 | 20.45 | 0.9985 | 0.3541 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-delta-1000` | `grouping_no_pruning_ablation` | yes | 200 | 53.07/53.81/54.15 | 19.49 | 18.82 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `delta_flat_full_scan` | yes | 200 | 24.02/24.25/24.51 | 45.58 | 41.67 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `delta_hnsw_ef128` | yes | 200 | 23.73/24/24.11 | 46.13 | 42.17 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `delta_hnsw_ef512` | yes | 200 | 24.08/24.4/24.51 | 45.49 | 41.54 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.96/17.18/21.15 | 61.48 | 58.64 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4942/0.598/1.918 | 2811 | 1940 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.195/1.826/3.206 | 924.8 | 781 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `group_pruning_beta0` | yes | 200 | 57.89/60.09/60.29 | 17.22 | 17.4 | 0.9985 | 0.04375 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 65.12/67.71/68.35 | 14.99 | 15.47 | 0.9985 | 0.04375 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `group_pruning_beta_factor_0p05` | yes | 200 | 57.8/60.15/60.5 | 17.37 | 17.49 | 0.9985 | 0.05813 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-16` | `grouping_no_pruning_ablation` | yes | 200 | 58.63/60.24/60.77 | 16.76 | 17.02 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `delta_flat_full_scan` | yes | 200 | 24.74/25.03/26.12 | 45.51 | 40.47 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `delta_hnsw_ef128` | yes | 200 | 24.53/24.85/25.71 | 46 | 40.81 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `delta_hnsw_ef512` | yes | 200 | 24.89/25.15/25.91 | 45.2 | 40.24 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.27/16.62/21.91 | 62.75 | 60.82 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4569/0.6297/2.109 | 2360 | 1985 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.151/1.608/3.019 | 1114 | 820.8 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `group_pruning_beta0` | yes | 200 | 189.4/217/220.3 | 5.321 | 5.265 | 0.9985 | 0.3091 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 193.6/221.2/228.1 | 5.151 | 5.149 | 0.9985 | 0.3091 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `group_pruning_beta_factor_0p05` | yes | 200 | 186.4/213.1/221.1 | 5.42 | 5.361 | 0.9985 | 0.3453 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-512` | `grouping_no_pruning_ablation` | yes | 200 | 219/226.5/227.9 | 4.541 | 4.538 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `delta_flat_full_scan` | yes | 200 | 24.72/25.05/25.92 | 44.59 | 40.47 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `delta_hnsw_ef128` | yes | 200 | 24.46/24.71/24.85 | 45.47 | 40.93 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `delta_hnsw_ef512` | yes | 200 | 24.8/25.11/25.39 | 44.58 | 40.34 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.88/17.15/17.24 | 61.63 | 58.96 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4787/0.581/0.8161 | 2719 | 2083 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.177/1.453/2.3 | 1070 | 823.4 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `group_pruning_beta0` | yes | 200 | 71.62/76.98/77.45 | 14.27 | 14.15 | 0.9985 | 0.1102 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 78.7/84.35/85.24 | 12.72 | 12.97 | 0.9985 | 0.1102 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `group_pruning_beta_factor_0p05` | yes | 200 | 70.67/76.36/77.01 | 14.46 | 14.39 | 0.9985 | 0.1388 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-groups-64` | `grouping_no_pruning_ablation` | yes | 200 | 74.94/77.56/78.8 | 13.24 | 13.24 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `certified_full_delta_reference` | yes | 1000 | N/A/N/A/N/A | N/A | N/A | 0.9989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `delta_flat_full_scan` | yes | 1000 | 22.41/23.69/23.91 | 46.52 | 44.12 | 0.9989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `delta_hnsw_ef128` | yes | 1000 | 22.17/23.46/23.63 | 46.65 | 44.59 | 0.9989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `delta_hnsw_ef32` | yes | 1000 | 22.05/23.34/23.47 | 47.19 | 44.82 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `delta_hnsw_ef512` | yes | 1000 | 22.51/23.8/23.99 | 46.21 | 43.91 | 0.9989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `full_base_delta_exact_flat` | yes | 1000 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `full_base_delta_flat_scan_performance` | yes | 1000 | 16.34/16.88/16.98 | 63.24 | 60.7 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `full_base_delta_hnsw_ef128` | yes | 1000 | 0.4244/0.514/0.556 | 2863 | 2395 | 0.9989 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `full_base_delta_hnsw_ef32` | yes | 1000 | 0.2117/0.2623/0.2836 | 5964 | 4730 | 0.9685 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `full_base_delta_hnsw_ef512` | yes | 1000 | 1.091/1.303/1.373 | 1058 | 928.1 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `group_pruning_beta0` | yes | 1000 | 81.79/89.97/91.42 | 12.21 | 12.43 | 0.9989 | 0.2192 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 1000 | 87.16/96.45/98.41 | 11.17 | 11.67 | 0.9989 | 0.2192 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `group_pruning_beta_factor_0p01` | yes | 1000 | 81.67/89.96/91.86 | 12.21 | 12.44 | 0.9989 | 0.2253 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `group_pruning_beta_factor_0p05` | yes | 1000 | 80.18/89.61/91.11 | 12.37 | 12.61 | 0.9989 | 0.2499 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `group_pruning_beta_factor_0p1` | yes | 1000 | 78.1/88.88/90.74 | 12.51 | 12.85 | 0.9989 | 0.283 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-0` | `grouping_no_pruning_ablation` | yes | 1000 | 90.4/92.58/93.5 | 11.04 | 11.05 | 0.9989 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `delta_flat_full_scan` | yes | 200 | 23.15/23.56/24.3 | 45.97 | 43.11 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `delta_hnsw_ef128` | yes | 200 | 22.92/23.31/23.87 | 46.61 | 43.57 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `delta_hnsw_ef512` | yes | 200 | 23.25/23.64/24.33 | 46.11 | 42.91 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.26/16.52/17.5 | 61.82 | 61.22 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.481/0.6218/1.288 | 3458 | 2004 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.211/1.692/2.14 | 846.2 | 811 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `group_pruning_beta0` | yes | 200 | 84.6/94.9/96.15 | 12.19 | 11.8 | 0.9985 | 0.1683 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 90.79/102.4/103.7 | 11.2 | 11.04 | 0.9985 | 0.1683 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `group_pruning_beta_factor_0p05` | yes | 200 | 83.51/94.61/96.05 | 12.4 | 11.97 | 0.9985 | 0.1994 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-1` | `grouping_no_pruning_ablation` | yes | 200 | 93.22/96.88/98.56 | 10.71 | 10.69 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `delta_flat_full_scan` | yes | 200 | 23.86/24.2/25.07 | 45.67 | 41.87 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `delta_hnsw_ef128` | yes | 200 | 23.64/24.18/24.93 | 45.78 | 42.23 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `delta_hnsw_ef512` | yes | 200 | 23.98/24.41/25.19 | 45.29 | 41.65 | 0.9985 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `full_base_delta_flat_scan_performance` | yes | 200 | 16.4/16.64/16.83 | 61.17 | 60.67 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4849/0.5883/0.9257 | 2743 | 2026 | 0.997 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.214/1.506/2.091 | 739.1 | 807 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `group_pruning_beta0` | yes | 200 | 87.95/94.94/96.01 | 11.46 | 11.64 | 0.9985 | 0.1753 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 92.22/101.9/103.6 | 10.52 | 10.96 | 0.9985 | 0.1753 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `group_pruning_beta_factor_0p05` | yes | 200 | 86.18/95.12/96.7 | 11.6 | 11.76 | 0.9985 | 0.2067 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-initial-group-build-seed-2` | `grouping_no_pruning_ablation` | yes | 200 | 94.86/96.93/97.51 | 10.57 | 10.54 | 0.9985 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.995 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `delta_flat_full_scan` | yes | 200 | 13.71/13.93/14.96 | 89.93 | 72.95 | 0.995 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `delta_hnsw_ef128` | yes | 200 | 13.48/13.68/14.79 | 91.58 | 74.27 | 0.995 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `delta_hnsw_ef512` | yes | 200 | 13.8/14.07/15.06 | 88.83 | 72.41 | 0.995 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `full_base_delta_flat_scan_performance` | yes | 200 | 5.399/5.587/5.642 | 208.3 | 183.1 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.4539/0.5742/0.9997 | 3114 | 2148 | 0.995 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.159/1.383/2.257 | 1142 | 847.8 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `group_pruning_beta0` | yes | 200 | 72.51/84.28/85.78 | 13.13 | 13.84 | 0.995 | 0.2645 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 77.72/89.97/92.08 | 11.87 | 13.02 | 0.995 | 0.2645 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `group_pruning_beta_factor_0p05` | yes | 200 | 71.51/83.4/86.17 | 13.35 | 14.04 | 0.995 | 0.2937 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-1` | `grouping_no_pruning_ablation` | yes | 200 | 85.04/87.88/88.32 | 11.61 | 11.69 | 0.995 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.9912 | 0 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `delta_flat_full_scan` | yes | 200 | 134.5/135.6/136.9 | 7.53 | 7.428 | 0.9912 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `delta_hnsw_ef128` | yes | 200 | 134.3/135.5/137 | 7.512 | 7.44 | 0.9912 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `delta_hnsw_ef512` | yes | 200 | 134.7/135.6/136.9 | 7.536 | 7.422 | 0.9912 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `full_base_delta_flat_scan_performance` | yes | 200 | 125.3/126.7/135.1 | 7.993 | 7.955 | 1 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.6728/0.9722/2.496 | 1669 | 1383 | 0.9872 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.391/2.882/4.449 | 863.8 | 648.7 | 0.9999 | N/A | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `group_pruning_beta0` | yes | 200 | 203/207.7/208.7 | 4.985 | 4.976 | 0.9912 | 0.1161 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 209.9/217.4/219.2 | 4.794 | 4.801 | 0.9912 | 0.1161 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `group_pruning_beta_factor_0p05` | yes | 200 | 201.6/209/210.4 | 5.002 | 4.994 | 0.9912 | 0.1459 | 0 | 0 |
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` / `sift-k-100` | `grouping_no_pruning_ablation` | yes | 200 | 207.3/210/210.9 | 4.797 | 4.817 | 0.9912 | 0 | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `certified_full_delta_reference` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 0.999 | 0 | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `delta_flat_full_scan` | yes | 200 | 27.62/32.76/38.12 | 38.33 | 35.24 | 0.999 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `delta_hnsw_ef128` | yes | 200 | 24.61/26.25/27.52 | 43.79 | 40.23 | 0.998 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `delta_hnsw_ef512` | yes | 200 | 25.74/28.13/30.02 | 42.09 | 38.4 | 0.999 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `full_base_delta_exact_flat` | yes | 200 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `full_base_delta_flat_scan_performance` | yes | 200 | 19.28/28.48/31.29 | 51.17 | 49.45 | 1 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `full_base_delta_hnsw_ef128` | yes | 200 | 0.6684/1.764/2.522 | 832 | 1237 | 0.998 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `full_base_delta_hnsw_ef512` | yes | 200 | 1.943/5.609/7.375 | 705.2 | 420.3 | 1 | N/A | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `group_pruning_beta0` | yes | 200 | 567.5/596.4/607.4 | 1.768 | 1.819 | 0.999 | 0.1173 | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 200 | 632.8/663.7/669.6 | 1.605 | 1.632 | 0.999 | 0.1173 | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `group_pruning_beta_factor_0p05` | yes | 200 | 555.1/594.5/609.2 | 1.791 | 1.859 | 0.999 | 0.1452 | 0 | 0 |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` / `sift-delta-100000` | `grouping_no_pruning_ablation` | yes | 200 | 594.4/605.4/636.4 | 1.649 | 1.684 | 0.999 | 0 | 0 | 0 |

## LB・tau・radius の保存分布

- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/gist-initial-group-build-seed-0`: non-timed audit LB n=3175, p50=0, p95=0.9518; same-C reference tau n=800, p50=1.215, p95=1.786; radius n=128, p50=1.692, p95=2.637
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/gist-initial-group-build-seed-1`: non-timed audit LB n=3200, p50=0, p95=1.114; same-C reference tau n=200, p50=1.206, p95=1.719; radius n=128, p50=1.707, p95=2.647
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/gist-initial-group-build-seed-2`: non-timed audit LB n=3150, p50=0, p95=0.938; same-C reference tau n=200, p50=1.206, p95=1.719; radius n=128, p50=1.696, p95=2.754
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-base-ef32`: non-timed audit LB n=3200, p50=152.7, p95=311.7; same-C reference tau n=200, p50=258.3, p95=305.4; radius n=128, p50=334.4, p95=380.8
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-base-ef512`: non-timed audit LB n=3200, p50=138, p95=313.5; same-C reference tau n=200, p50=258.1, p95=305; radius n=128, p50=334.4, p95=380.8
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-delta-0`: non-timed audit LB n=0, p50=N/A, p95=N/A; same-C reference tau n=200, p50=258.1, p95=305; radius n=128, p50=0, p95=0
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-delta-1000`: non-timed audit LB n=3200, p50=197, p95=354.4; same-C reference tau n=200, p50=258.1, p95=305; radius n=128, p50=279.8, p95=332.8
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-groups-16`: non-timed audit LB n=400, p50=91.58, p95=235.9; same-C reference tau n=200, p50=258.1, p95=305; radius n=16, p50=404.1, p95=452.4
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-groups-512`: non-timed audit LB n=12800, p50=202.2, p95=354; same-C reference tau n=200, p50=258.1, p95=305; radius n=512, p50=291.5, p95=340.6
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-groups-64`: non-timed audit LB n=1600, p50=124.2, p95=272.2; same-C reference tau n=200, p50=258.1, p95=305; radius n=64, p50=361.6, p95=395.7
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-initial-group-build-seed-0`: non-timed audit LB n=3200, p50=144, p95=319.6; same-C reference tau n=1000, p50=250.6, p95=302.8; radius n=128, p50=334.4, p95=380.8
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-initial-group-build-seed-1`: non-timed audit LB n=3200, p50=149.1, p95=309.5; same-C reference tau n=200, p50=258.1, p95=305; radius n=128, p50=334.6, p95=383.5
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-initial-group-build-seed-2`: non-timed audit LB n=3200, p50=158.6, p95=319.9; same-C reference tau n=200, p50=258.1, p95=305; radius n=128, p50=339.9, p95=381
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-k-1`: non-timed audit LB n=3200, p50=147.3, p95=301.1; same-C reference tau n=200, p50=229.4, p95=280.5; radius n=128, p50=334.4, p95=380.8
- `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0/sift-k-100`: non-timed audit LB n=3200, p50=145.9, p95=312; same-C reference tau n=200, p50=288.7, p95=333.7; radius n=128, p50=334.4, p95=380.8
- `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085/sift-delta-100000`: non-timed audit LB n=3200, p50=121.7, p95=275.9; same-C reference tau n=200, p50=252, p95=296.5; radius n=128, p50=365.1, p95=399.4

## 読み方と制約

- coupled Delta micro は凍結済み base 候補 `C` の取得時間を除外し、E2E はその取得時間を加算する。全体 HNSW / 全体 exact Flat は E2E のみである。
- Faiss の L2 index が返す二乗距離は通常 L2 に変換して保存した。保証付き方式の observed beta は保存 float32 の k 番目距離を Fraction で再評価した区間上限で判定した。独立 oracle の全順位照合は manifest に保存された決定的sampleだけであり、population上限によるskipも隠さない。
- `実測batch QPS` は artifact書込みとtruth検証を除いた単一thread逐次batchの wall time。`単query容量推定QPS` は queryごとの service time 平均の逆数であり、実throughputではない。主percentileはrepeatをqueryごとにmedian化してから集計し、pooled repeat p99はsummaryの補助列にのみ残す。
- `certified_full` と `full_exact_truth` は queryごとに1回だけ生成する検証用truthであり、性能比較対象から除外した。その生成wall timeは summary の `truth_generation_wall_ms` に別保存した。
- HNSW の per-query visited vector 数は利用中の Faiss API から信頼できる形で取得できないため `null`。構築/add は ACID 更新性能ではなく component cost である。
- 1,000 query の p99 も有限標本であり、共有ホストの負荷変動を含む。1,000 未満の条件には summary の警告列が付く。
- Delta が same-C 基準 top-k に入る query の割合、当該部分集合の latency/recall は summary JSON/CSV に保存した。
- この自動素材だけから画像 descriptor の結果を RAG 全般へ外挿しない。

## 図

- `reports/figures/latency_quantiles.png`
- `reports/figures/quality_latency.png`
- `reports/figures/group_scan_skip.png`

最終の `SUPPORTED_IN_TESTED_REGIME` / `NOT_SUPPORTED_IN_TESTED_REGIME` / `INCONCLUSIVE` 判定は、完了 run、契約違反、維持費、実データ範囲、negative result を併せて人が行う。
