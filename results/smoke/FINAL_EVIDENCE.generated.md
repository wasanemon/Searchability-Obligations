# 実験生データからの自動集計（最終報告データ節素材）

生成時刻 (UTC): `2026-09-05T12:14:23.299454+00:00`

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
| `offline-smoke-20260905T121406.178089Z-43d3397eca` | yes | `43d3397eca1f74b736d7e0b94184e900ca879f55d4d2c1f5ab206d8492f5f4bc` | synthetic-clustered: 281859744808 / aa18c94a5199; synthetic-delta-near-queries: 20b08b6cbdc7 / 3ef146be427a; synthetic-isotropic: d12bb7cf450b / d15900207832; synthetic-outlier-radius: bffb4b5356f2 / eeef9255dbe2 | completed |

## 方式別集計

| run / 条件 | 方式 | 完了 | query | E2E p50/p95/p99 ms | 実測batch QPS p50 | 単query容量推定QPS | exact recall@k 平均 | skip率 | fallback率 | 違反 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `delta_flat_full_scan` | yes | 8 | 1.68/1.798/1.824 | 611.6 | 587.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `delta_hnsw_ef24` | yes | 8 | 1.681/1.74/1.744 | 609.8 | 589.2 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `delta_hnsw_ef8` | yes | 8 | 1.694/1.764/1.771 | 606.7 | 584.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.306/1.361/1.379 | 770.6 | 760.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07115/0.07446/0.07454 | 1.452e+04 | 1.475e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06717/0.07372/0.07427 | 1.437e+04 | 1.474e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta0` | yes | 8 | 2.659/2.824/2.84 | 354.5 | 372.1 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.746/2.882/2.891 | 340.9 | 361.8 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.657/2.802/2.811 | 349.1 | 374.6 | 1 | 0.5625 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-clustered` | `grouping_no_pruning_ablation` | yes | 8 | 2.836/2.898/2.903 | 334.2 | 352.1 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_flat_full_scan` | yes | 8 | 1.699/1.792/1.805 | 609.5 | 585.1 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_hnsw_ef24` | yes | 8 | 1.707/1.743/1.746 | 603.9 | 584.9 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `delta_hnsw_ef8` | yes | 8 | 1.68/1.795/1.803 | 606.3 | 585.5 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.312/1.341/1.346 | 762 | 760.3 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07227/0.09047/0.096 | 1.529e+04 | 1.349e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06668/0.07543/0.07581 | 1.367e+04 | 1.474e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta0` | yes | 8 | 2.458/2.49/2.496 | 380.9 | 407.3 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.535/2.662/2.68 | 370.4 | 391.6 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.448/2.494/2.5 | 381.7 | 407.3 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-delta-near-queries` | `grouping_no_pruning_ablation` | yes | 8 | 2.453/2.506/2.509 | 379.1 | 407.1 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `delta_flat_full_scan` | yes | 8 | 2.129/2.196/2.197 | 478.3 | 466.5 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `delta_hnsw_ef24` | yes | 8 | 2.139/2.244/2.268 | 475.9 | 463.8 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `delta_hnsw_ef8` | yes | 8 | 2.142/2.2/2.211 | 475.2 | 465.6 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.76/1.783/1.785 | 566.2 | 567 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07287/0.07874/0.07992 | 1.348e+04 | 1.392e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.07258/0.07668/0.07751 | 1.421e+04 | 1.387e+04 | 0.925 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta0` | yes | 8 | 3.278/3.344/3.348 | 291 | 304.3 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 3.379/3.412/3.414 | 282.7 | 296.5 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `group_pruning_beta_factor_0p05` | yes | 8 | 3.289/3.418/3.442 | 289.6 | 302 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-isotropic` | `grouping_no_pruning_ablation` | yes | 8 | 3.294/3.346/3.359 | 291 | 303.5 | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `certified_full_delta_reference` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `delta_flat_full_scan` | yes | 8 | 1.678/1.764/1.768 | 614 | 589.7 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `delta_hnsw_ef24` | yes | 8 | 1.702/1.768/1.774 | 607.4 | 584.6 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `delta_hnsw_ef8` | yes | 8 | 1.696/1.804/1.828 | 605.1 | 582.7 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_exact_flat` | yes | 8 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_flat_scan_performance` | yes | 8 | 1.32/1.342/1.343 | 768.6 | 759.7 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_hnsw_ef24` | yes | 8 | 0.07272/0.07683/0.07713 | 1.355e+04 | 1.407e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `full_base_delta_hnsw_ef8` | yes | 8 | 0.06559/0.07178/0.0723 | 1.394e+04 | 1.502e+04 | 1 | N/A | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta0` | yes | 8 | 2.545/2.612/2.616 | 365.8 | 390.7 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta0_recompute_intervals_ablation` | yes | 8 | 2.638/2.698/2.706 | 354.6 | 377.1 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `group_pruning_beta_factor_0p05` | yes | 8 | 2.584/2.688/2.7 | 364 | 384.2 | 1 | 0.75 | 0 | 0 |
| `offline-smoke-20260905T121406.178089Z-43d3397eca` / `synthetic-outlier-radius` | `grouping_no_pruning_ablation` | yes | 8 | 2.819/2.91/2.929 | 334.3 | 353 | 1 | 0 | 0 | 0 |

## LB・tau・radius の保存分布

- `offline-smoke-20260905T121406.178089Z-43d3397eca/synthetic-clustered`: non-timed audit LB n=16, p50=2.319, p95=50.62; same-C reference tau n=8, p50=3.112, p95=3.414; radius n=4, p50=3.229, p95=35.82
- `offline-smoke-20260905T121406.178089Z-43d3397eca/synthetic-delta-near-queries`: non-timed audit LB n=8, p50=0, p95=1.102; same-C reference tau n=8, p50=3.388, p95=4.051; radius n=4, p50=1.601, p95=4.219
- `offline-smoke-20260905T121406.178089Z-43d3397eca/synthetic-isotropic`: non-timed audit LB n=16, p50=0.09508, p95=0.9791; same-C reference tau n=8, p50=5.567, p95=6.035; radius n=4, p50=5.445, p95=5.926
- `offline-smoke-20260905T121406.178089Z-43d3397eca/synthetic-outlier-radius`: non-timed audit LB n=16, p50=10.93, p95=18.9; same-C reference tau n=8, p50=2.023, p95=2.371; radius n=4, p50=6.99, p95=14.89

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

- `results/smoke/final_figures/latency_quantiles.png`
- `results/smoke/final_figures/quality_latency.png`
- `results/smoke/final_figures/group_scan_skip.png`

最終の `SUPPORTED_IN_TESTED_REGIME` / `NOT_SUPPORTED_IN_TESTED_REGIME` / `INCONCLUSIVE` 判定は、完了 run、契約違反、維持費、実データ範囲、negative result を併せて人が行う。
