# 実験生データからの自動集計（最終報告データ節素材）

生成時刻 (UTC): `2026-09-05T06:27:07.365687+00:00`

この文書は保存済み JSONL と manifest のみから再生成した集計素材であり、研究仮説の最終判断を自動で行わない。`final` 指定の未完了 run は集計から fail-closed で除外し、下表に状態を残す。その他の未完了 run は `完了` 列で区別する。

## 入力と完全性

- run 数: 1（完了 marker あり: 1）
- 未完了のため除外した final run: 0
- query-method 生行数: 384
- 契約違反として保存された行数: 0
- optimized baseline 検証失敗行数: 0
- 独立 Fraction oracle: 実行 4 / mismatch 0 / 上限による skip 0
- raw shard は checkpoint 記載の SHA-256 と行数を再検証済み。検索処理は分析時に再実行していない。

### Run 識別子

| run ID | 完了 | config hash | dataset hash / split ID | 状態 |
|---|---:|---|---|---|
| `offline-smoke-20260905T062705.177285Z-43d3397eca` | yes | `43d3397eca1f74b736d7e0b94184e900ca879f55d4d2c1f5ab206d8492f5f4bc` | synthetic-clustered: 281859744808 / aa18c94a5199; synthetic-delta-near-queries: 20b08b6cbdc7 / 3ef146be427a; synthetic-isotropic: d12bb7cf450b / d15900207832; synthetic-outlier-radius: bffb4b5356f2 / eeef9255dbe2 | completed |

## 方式別集計

| run / 条件 | 方式 | 完了 | query | E2E p50/p95/p99 ms | 実測batch QPS p50 | 単query容量推定QPS | exact recall@k 平均 | skip率 | fallback率 | 違反 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `delta_flat_full_scan` | yes | 8 | 1.672/1.779/1.81 | 592 | 591.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `delta_hnsw_ef24` | yes | 8 | 1.683/1.727/1.728 | 615.6 | 593.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `delta_hnsw_ef8` | yes | 8 | 1.68/1.744/1.745 | 612.8 | 590.2 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.3/1.357/1.371 | 773.9 | 764.9 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.06993/0.07427/0.07515 | 1.488e+04 | 1.493e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06643/0.07199/0.07268 | 1.465e+04 | 1.504e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta0` | yes | 8 | 2.645/2.789/2.817 | 355.9 | 375.1 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.744/2.841/2.847 | 344.7 | 364.3 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.638/2.765/2.772 | 352.1 | 377.4 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-clustered` | `grouping_no_pruning_ablation` | yes | 8 | 2.821/2.874/2.883 | 336.5 | 354 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_flat_full_scan` | yes | 8 | 1.658/1.773/1.792 | 614.2 | 594.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_hnsw_ef24` | yes | 8 | 1.681/1.729/1.732 | 613.8 | 592.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_hnsw_ef8` | yes | 8 | 1.658/1.747/1.749 | 614.1 | 596.1 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.307/1.311/1.312 | 767.8 | 768.1 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07215/0.07613/0.07636 | 1.571e+04 | 1.421e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06448/0.07187/0.07188 | 1.407e+04 | 1.525e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta0` | yes | 8 | 2.423/2.507/2.52 | 383.5 | 409.7 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.5/2.639/2.657 | 372.7 | 396.1 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.428/2.485/2.488 | 383.2 | 410.6 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-delta-near-queries` | `grouping_no_pruning_ablation` | yes | 8 | 2.441/2.502/2.51 | 383.1 | 409.4 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `delta_flat_full_scan` | yes | 8 | 2.112/2.192/2.199 | 482.8 | 470.1 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `delta_hnsw_ef24` | yes | 8 | 2.119/2.226/2.254 | 479 | 468.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `delta_hnsw_ef8` | yes | 8 | 2.119/2.192/2.195 | 483.2 | 469 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.743/1.766/1.77 | 573 | 573.9 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07084/0.08018/0.08052 | 1.389e+04 | 1.4e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06996/0.07099/0.07112 | 1.501e+04 | 1.455e+04 | 0.925 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta0` | yes | 8 | 3.26/3.317/3.32 | 293.6 | 306.7 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 3.339/3.391/3.393 | 284.9 | 299.4 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta_factor_0p05` | yes | 8 | 3.269/3.38/3.406 | 290.8 | 304.2 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-isotropic` | `grouping_no_pruning_ablation` | yes | 8 | 3.276/3.316/3.329 | 292.8 | 306.1 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `delta_flat_full_scan` | yes | 8 | 1.66/1.743/1.746 | 617.5 | 596.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `delta_hnsw_ef24` | yes | 8 | 1.694/1.747/1.756 | 609.1 | 590.2 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `delta_hnsw_ef8` | yes | 8 | 1.681/1.785/1.81 | 613.6 | 588.4 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.299/1.334/1.34 | 771.6 | 765.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07213/0.07665/0.07774 | 1.396e+04 | 1.423e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06452/0.07265/0.0727 | 1.436e+04 | 1.522e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta0` | yes | 8 | 2.534/2.619/2.621 | 366.4 | 391.9 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.632/2.7/2.704 | 353.9 | 377.9 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.559/2.666/2.684 | 364.4 | 387.1 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T062705.177285Z-43d3397eca` / `synthetic-outlier-radius` | `grouping_no_pruning_ablation` | yes | 8 | 2.814/2.882/2.904 | 337.6 | 355.3 | 1 | 0 | 0 | 0 |

## LB と radius の保存分布

- `offline-smoke-20260905T062705.177285Z-43d3397eca/synthetic-clustered`: LB n=16, p50=2.319, p95=50.62; radius n=4, p50=3.229, p95=35.82
- `offline-smoke-20260905T062705.177285Z-43d3397eca/synthetic-delta-near-queries`: LB n=8, p50=0, p95=1.102; radius n=4, p50=1.601, p95=4.219
- `offline-smoke-20260905T062705.177285Z-43d3397eca/synthetic-isotropic`: LB n=16, p50=0.09508, p95=0.9791; radius n=4, p50=5.445, p95=5.926
- `offline-smoke-20260905T062705.177285Z-43d3397eca/synthetic-outlier-radius`: LB n=16, p50=10.93, p95=18.9; radius n=4, p50=6.99, p95=14.89

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
