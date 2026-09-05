# 実験生データからの自動集計（最終報告データ節素材）

生成時刻 (UTC): `2026-09-05T06:06:41.598200+00:00`

この文書は保存済み JSONL と manifest のみから再生成した集計素材であり、研究仮説の最終判断を自動で行わない。未完了 run の行は `完了` 列で区別する。

## 入力と完全性

- run 数: 1（完了 marker あり: 1）
- query-method 生行数: 75
- 契約違反として保存された行数: 0
- optimized baseline 検証失敗行数: 0
- 独立 Fraction oracle: 実行 1 / mismatch 0 / 上限による skip 0
- raw shard は checkpoint 記載の SHA-256 と行数を再検証済み。検索処理は分析時に再実行していない。

### Run 識別子

| run ID | 完了 | config hash | dataset hash / split ID | 状態 |
|---|---:|---|---|---|
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` | yes | `72c43841b0a3c2580069affdcee83755a29303780ad381cf72fa07b570705c85` | sift-initial-build-seed-0: 939f2d07b551 / 0f66d15fa9e1 | completed |

## 方式別集計

| run / 条件 | 方式 | 完了 | query | E2E p50/p95/p99 ms | 実測batch QPS p50 | 単query容量推定QPS | exact recall@k 平均 | skip率 | fallback率 | 違反 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `certified_full_delta_reference` | yes | 5 | N/A/N/A/N/A | N/A | N/A | 1 | 0 | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `delta_flat_full_scan` | yes | 5 | 13.7/13.82/13.83 | 74.3 | 72.9 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `delta_hnsw_ef128` | yes | 5 | 13.76/13.81/13.82 | 73.76 | 72.82 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `delta_hnsw_ef32` | yes | 5 | 13.68/13.77/13.79 | 74.21 | 73.02 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `delta_hnsw_ef512` | yes | 5 | 13.8/13.81/13.81 | 73.94 | 72.62 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `full_base_delta_exact_flat` | yes | 5 | N/A/N/A/N/A | N/A | N/A | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `full_base_delta_flat_scan_performance` | yes | 5 | 12.69/12.82/12.84 | 78.24 | 78.59 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `full_base_delta_hnsw_ef128` | yes | 5 | 0.1624/0.1733/0.1748 | 4164 | 6109 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `full_base_delta_hnsw_ef32` | yes | 5 | 0.09031/0.1168/0.1181 | 8438 | 1.022e+04 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `full_base_delta_hnsw_ef512` | yes | 5 | 0.3142/0.3412/0.3448 | 3536 | 3209 | 1 | N/A | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `group_pruning_beta0` | yes | 5 | 25.65/26.55/26.63 | 38.05 | 38.91 | 1 | 0.3046 | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `group_pruning_beta_factor_0p01` | yes | 5 | 25.43/26.69/26.7 | 38.07 | 38.87 | 1 | 0.3262 | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `group_pruning_beta_factor_0p05` | yes | 5 | 25.27/26.63/26.79 | 38.24 | 39.23 | 1 | 0.3631 | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `group_pruning_beta_factor_0p1` | yes | 5 | 24.94/26.29/26.46 | 38.69 | 39.67 | 1 | 0.4062 | 0 | 0 |
| `texmex-evaluation-20260905T055941.965056Z-72c43841b0` / `sift-initial-build-seed-0` | `grouping_no_pruning_ablation` | yes | 5 | 27.21/27.38/27.38 | 35.98 | 36.69 | 1 | 0 | 0 | 0 |

## LB と radius の保存分布

- `texmex-evaluation-20260905T055941.965056Z-72c43841b0/sift-initial-build-seed-0`: LB n=325, p50=233.9, p95=383.1; radius n=128, p50=145.4, p95=308.3

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
