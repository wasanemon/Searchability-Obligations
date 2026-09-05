# Native kernel 再検証報告（Issue #3）

研究全体の事前規定 stopping-rule verdict: **`NOT_SUPPORTED_IN_TESTED_REGIME`**、工学的導入判定: **`NO_GO`**（validation gate: **NOT_PASSED**、final status: **`NOT_RUN_GATE_NOT_PASSED`**）。これは fresh-final holdout estimate ではない。validation gate は通過しなかったため、規定どおり大規模 final holdout は実行していない。

Validation gate の保存理由: `no_non_degenerate_candidate_beats_both_F_and_A`。

本報告は [GitHub Issue #3](https://github.com/wasanemon/Searchability-Obligations/issues/3) の手順に従い、旧 Python 実装の遅さと center-radius pruning 自体の限界を分離して再検証した結果である。ordinary L2 と Faiss の squared-L2 を区別し、全 native 保証経路は [`docs/native_numerics.md`](../docs/native_numerics.md) の interval/error-bound と厳密境界順序を用いた。

## 結論（RQ1 / RQ2 / RQ3）

- **RQ1（P 対 F、同一保証）**: 事前指定 main `sift-initial` の F/P geomean 範囲は 1.097–1.131（95% CI lower > 1: 4/4）。gate-eligible SIFT 候補 16 件中、F/P criterion 通過は 16 件。real/non-degenerate の記述対象は計 18 件（GIST anchor 2 件を含み、その F/P 通過は 0/2）。secondary sweep 上の事後的最大 F/P=1.160 [CI 1.147, 1.174] (`sift-k-1` / `native_P_factor_0p05`)は記述のみで、事前指定 main family の確認的要約とは区別する（事前登録済みのgate candidate 集合には含まれる）。validation gate は `NOT_PASSED` で fresh final を許可しなかった。`selected_candidate` は `null`。 P と F は同じ frozen `C`・同じ保証で、`F/P` paired API wall が直接比較である。
- **RQ2（P 対 A、practical comparator）**: 事前指定 main `sift-initial` の A/P geomean 範囲は 0.579–0.597（95% CI upper < 1: 4/4）。gate-eligible SIFT 候補 16 件中、A/P timing+quality criterion 通過は 0 件。GIST anchor のA/P 通過は 0/2。secondary sweep 上の事後的最大 A/P=0.838 [CI 0.836, 0.840] (`sift-delta-1000` / `native_P_factor_0p05`)は記述のみで、事前指定 main family の確認的要約とは区別する（事前登録済みのgate candidate 集合には含まれる）。gate-eligible SIFT の A mismatch は 0 rows / 0 queries（timing から除外せず保持）。 A は optimized Faiss Delta Flat + small exact rerank だが、rerank は返却 shortlist 内だけで全 Delta に対する保証ではない。
- **RQ3（勝敗理由）**: 事前指定 main `sift-initial` 内（各 operating pointの session-query n=200）で N/P geomean 範囲=1.108–1.143、P の operating-point別 median skipped groups=20.0–30.0/128。scan gap median/p95=325.541/381.721 L2、skip gap median/p95=304.316/360.617 L2。incremental group build median=2728.653 ms、packed view を F/P 共通と扱う group-only の F 比有限 break-even は 4/4 operating points（2569.1–3512.8 queries）、packed build も P に課す感度分析は 4/4 points（2616.3–3577.4 queries）。他 dataset/secondary axis は層別表の記述値とし、異なる次元・座標尺度のL2 gap を pooled aggregate しない。これは保存 evidence の記述的対応であり、単独では勝敗原因を因果確定しない。 skip/read、N/P、ablation、Base cache、build・memory・限定的 break-even を併記して解釈する。

## 正しさ gate

- fixed-seed native cases: 10000。
- pytest JUnit: executed 255 passed、failures 0、errors 0、skipped 0、deselected bound JSON/JUnit に未保存（数値は主張しない）。
- native backend: `pybind11_cpp17`、shared object SHA-256 `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a`。
- compile flags: `['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off']`。fast-math 無効、FE_TONEAREST 強制、strict `LB > tau-beta`、曖昧境界だけ exact Fraction で再順位付けした。
- raw native 行の backend/call evidence と beta chain の aggregate failure はそれぞれ 0 / 0。
- Python fallback は F/N/P raw 行 0/25872、そのうち P は 0/13536。

## Validation 結果

latency は API wall。各 session/query 内で repetition median、session 間で比の幾何平均、query を独立単位として固定 seed の paired bootstrap（5000 resamples）を用いた。speedup は baseline/P なので 1 より大きいほど P が速い。A の品質不一致行も timing から除外していない。`n q/session-q` は paired query 数 / paired process-session-query observation 数である。

| condition | P method | beta (L2) | P p50/p95/p99 ms | F/P geomean [95% CI] | A/P geomean [95% CI] | N/P geomean | median skipped groups / vectors scanned | n q/session-q (F; A) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gist-initial | native_P_beta0 | 0.000000 | 28.892/48.973/49.362 | 0.995 [0.993, 0.998] | 0.557 [0.552, 0.562] | 1.008 | 2.0 / 4904.0 | F 200/200; A 200/200 |
| gist-initial | native_P_factor_0p05 | 0.057683 | 28.868/48.822/49.306 | 0.996 [0.993, 0.999] | 0.557 [0.552, 0.563] | 1.009 | 2.0 / 4904.0 | F 200/200; A 200/200 |
| sift-delta-0 | native_P_beta0 | 0.000000 | 5.026/5.120/6.178 | 1.001 [0.999, 1.002] | 0.854 [0.853, 0.856] | 1.000 | 0.0 / 0.0 | F 200/200; A 200/200 |
| sift-delta-0 | native_P_factor_0p05 | 12.454367 | 5.025/5.140/6.199 | 1.000 [0.998, 1.002] | 0.854 [0.852, 0.855] | 1.000 | 0.0 / 0.0 | F 200/200; A 200/200 |
| sift-delta-1000 | native_P_beta0 | 0.000000 | 5.443/5.586/6.603 | 1.006 [1.004, 1.008] | 0.835 [0.833, 0.836] | 1.020 | 41.0 / 677.5 | F 200/200; A 200/200 |
| sift-delta-1000 | native_P_factor_0p05 | 12.454367 | 5.419/5.556/6.617 | 1.010 [1.008, 1.012] | 0.838 [0.836, 0.840] | 1.024 | 47.5 / 635.0 | F 200/200; A 200/200 |
| sift-delta-100000 | native_P_beta0 | 0.000000 | 44.111/47.418/47.581 | 1.120 [1.102, 1.138] | 0.200 [0.197, 0.204] | 1.125 | 10.0 / 91813.5 | F 200/200; A 200/200 |
| sift-delta-100000 | native_P_factor_0p05 | 12.084539 | 42.778/47.436/47.541 | 1.153 [1.132, 1.175] | 0.206 [0.202, 0.210] | 1.158 | 14.0 / 88612.0 | F 200/200; A 200/200 |
| sift-groups-512 | native_P_beta0 | 0.000000 | 8.200/9.450/9.590 | 1.136 [1.124, 1.148] | 0.598 [0.592, 0.604] | 1.177 | 163.0 / 6384.5 | F 200/200; A 200/200 |
| sift-groups-512 | native_P_factor_0p05 | 12.405425 | 8.098/9.208/9.496 | 1.155 [1.143, 1.167] | 0.608 [0.602, 0.614] | 1.196 | 186.0 / 6058.0 | F 200/200; A 200/200 |
| sift-groups-64 | native_P_beta0 | 0.000000 | 8.991/9.450/10.239 | 1.058 [1.049, 1.067] | 0.558 [0.553, 0.563] | 1.064 | 4.0 / 8913.0 | F 200/200; A 200/200 |
| sift-groups-64 | native_P_factor_0p05 | 12.405425 | 8.878/9.473/10.076 | 1.074 [1.064, 1.084] | 0.566 [0.561, 0.572] | 1.080 | 6.0 / 8731.0 | F 200/200; A 200/200 |
| sift-initial | native_P_beta0 | 0.000000 | 8.516/9.289/9.723 | 1.097 [1.087, 1.108] | 0.579 [0.573, 0.585] | 1.108 | 20.0 / 7892.5 | F 200/200; A 200/200 |
| sift-initial | native_P_factor_0p01 | 2.481085 | 8.514/9.301/9.668 | 1.099 [1.089, 1.110] | 0.580 [0.574, 0.586] | 1.111 | 21.0 / 7796.0 | F 200/200; A 200/200 |
| sift-initial | native_P_factor_0p05 | 12.405425 | 8.371/9.244/9.679 | 1.114 [1.103, 1.126] | 0.588 [0.582, 0.594] | 1.125 | 24.5 / 7556.0 | F 200/200; A 200/200 |
| sift-initial | native_P_factor_0p1 | 24.810850 | 8.216/9.218/9.627 | 1.131 [1.120, 1.143] | 0.597 [0.591, 0.604] | 1.143 | 30.0 / 7139.0 | F 200/200; A 200/200 |
| sift-k-1 | native_P_beta0 | 0.000000 | 8.101/9.133/9.253 | 1.145 [1.131, 1.159] | 0.605 [0.597, 0.612] | 1.157 | 31.5 / 6973.5 | F 200/200; A 200/200 |
| sift-k-1 | native_P_factor_0p05 | 10.489454 | 7.950/9.013/9.282 | 1.160 [1.147, 1.174] | 0.613 [0.606, 0.620] | 1.172 | 38.5 / 6525.5 | F 200/200; A 200/200 |
| sift-k-100 | native_P_beta0 | 0.000000 | 10.865/13.648/15.411 | 1.047 [1.039, 1.054] | 0.578 [0.570, 0.586] | 1.055 | 9.5 / 8824.0 | F 200/200; A 200/200 |
| sift-k-100 | native_P_factor_0p05 | 13.900270 | 10.761/13.544/15.321 | 1.059 [1.052, 1.067] | 0.585 [0.577, 0.593] | 1.068 | 14.0 / 8426.0 | F 200/200; A 200/200 |
| synthetic-clustered | native_P_beta0 | 0.000000 | 1.833/1.873/1.883 | 0.991 [0.988, 0.995] | 0.672 [0.669, 0.674] | 0.994 | 14.0 / 18.0 | F 64/64; A 64/64 |
| synthetic-clustered | native_P_factor_0p05 | 0.155362 | 1.830/1.876/1.890 | 0.992 [0.988, 0.995] | 0.672 [0.670, 0.674] | 0.994 | 14.0 / 18.0 | F 64/64; A 64/64 |
| synthetic-delta-near-queries | native_P_beta0 | 0.000000 | 1.835/1.868/1.889 | 0.997 [0.993, 1.001] | 0.678 [0.676, 0.680] | 1.001 | 0.0 / 128.0 | F 64/64; A 64/64 |
| synthetic-delta-near-queries | native_P_factor_0p05 | 0.187547 | 1.834/1.875/1.883 | 0.996 [0.992, 1.001] | 0.677 [0.675, 0.680] | 1.001 | 0.0 / 128.0 | F 64/64; A 64/64 |
| synthetic-high-dimensional-isotropic | native_P_beta0 | 0.000000 | 3.230/3.270/3.278 | 0.994 [0.991, 0.996] | 0.803 [0.801, 0.805] | 0.997 | 0.0 / 128.0 | F 64/64; A 64/64 |
| synthetic-high-dimensional-isotropic | native_P_factor_0p05 | 0.703947 | 3.224/3.265/3.269 | 0.996 [0.993, 0.998] | 0.805 [0.803, 0.807] | 1.000 | 0.0 / 128.0 | F 64/64; A 64/64 |
| synthetic-outlier-radius | native_P_beta0 | 0.000000 | 1.830/1.855/1.861 | 0.993 [0.989, 0.997] | 0.673 [0.671, 0.675] | 0.996 | 10.0 / 42.0 | F 64/64; A 64/64 |
| synthetic-outlier-radius | native_P_factor_0p05 | 0.089763 | 1.826/1.856/1.862 | 0.994 [0.990, 0.997] | 0.673 [0.671, 0.675] | 0.996 | 10.0 / 42.0 | F 64/64; A 64/64 |

主条件の beta=0 について、3 timing scope を同じ分母で並べる。この validation は process session 1 回、各 query の session 数 1–1、paired query/session-query は 200/200である。したがって CI は query sampling の不確実性は表すが、process/session 間変動は推定しない。

| scope | F p50/p95/p99 ms | A p50/p95/p99 ms | P(beta=0) p50/p95/p99 ms | F/P geomean [95% CI] | A/P geomean [95% CI] | n q/session-q |
|---|---:|---:|---:|---:|---:|---:|
| micro | 6.300/6.398/7.512 | 2.220/2.263/3.523 | 5.594/6.388/6.814 | 1.151 [1.135, 1.169] | 0.408 [0.402, 0.415] | 200/200 |
| composed_e2e | 9.879/10.072/11.007 | 5.425/5.558/6.654 | 9.172/9.992/10.309 | 1.090 [1.080, 1.101] | 0.600 [0.595, 0.606] | 200/200 |
| api_wall | 9.241/9.378/10.424 | 4.866/4.966/6.053 | 8.516/9.289/9.723 | 1.097 [1.087, 1.108] | 0.579 [0.573, 0.585] | 200/200 |

## Delta-influence subset

Delta neighbor が exact visible top-k に影響する query だけを同じ query-paired bootstrap で再集計した。下表は real/non-degenerate かつ SIFT primary で subset n>0 の gate-eligible 候補を全件（n=16）表示する。subset CI 下端 > 1 は F/P で 15/16、A/P で 0/16（A/P の CI 上端 < 1 は 16/16）。

| condition | P method | beta (L2) | influenced queries n | F/P subset geomean [95% CI] | A/P subset geomean [95% CI] |
|---|---|---:|---:|---:|---:|
| sift-delta-1000 | native_P_beta0 | 0.000000 | 16 | 1.006 [1.000, 1.014] | 0.837 [0.830, 0.845] |
| sift-delta-1000 | native_P_factor_0p05 | 12.454367 | 16 | 1.008 [1.001, 1.016] | 0.838 [0.831, 0.847] |
| sift-delta-100000 | native_P_beta0 | 0.000000 | 197 | 1.118 [1.100, 1.137] | 0.200 [0.196, 0.203] |
| sift-delta-100000 | native_P_factor_0p05 | 12.084539 | 197 | 1.151 [1.130, 1.173] | 0.206 [0.202, 0.209] |
| sift-groups-512 | native_P_beta0 | 0.000000 | 101 | 1.140 [1.124, 1.156] | 0.601 [0.592, 0.610] |
| sift-groups-512 | native_P_factor_0p05 | 12.405425 | 101 | 1.158 [1.142, 1.174] | 0.610 [0.601, 0.619] |
| sift-groups-64 | native_P_beta0 | 0.000000 | 101 | 1.065 [1.052, 1.078] | 0.561 [0.554, 0.568] |
| sift-groups-64 | native_P_factor_0p05 | 12.405425 | 101 | 1.082 [1.068, 1.098] | 0.570 [0.562, 0.578] |
| sift-initial | native_P_beta0 | 0.000000 | 101 | 1.105 [1.090, 1.121] | 0.583 [0.575, 0.591] |
| sift-initial | native_P_factor_0p01 | 2.481085 | 101 | 1.107 [1.091, 1.122] | 0.584 [0.576, 0.592] |
| sift-initial | native_P_factor_0p05 | 12.405425 | 101 | 1.120 [1.103, 1.136] | 0.591 [0.582, 0.600] |
| sift-initial | native_P_factor_0p1 | 24.810850 | 101 | 1.138 [1.121, 1.156] | 0.600 [0.592, 0.609] |
| sift-k-1 | native_P_beta0 | 0.000000 | 12 | 1.164 [1.119, 1.215] | 0.612 [0.588, 0.635] |
| sift-k-1 | native_P_factor_0p05 | 10.489454 | 12 | 1.179 [1.134, 1.227] | 0.620 [0.599, 0.643] |
| sift-k-100 | native_P_beta0 | 0.000000 | 200 | 1.047 [1.040, 1.054] | 0.578 [0.570, 0.585] |
| sift-k-100 | native_P_factor_0p05 | 13.900270 | 200 | 1.059 [1.052, 1.067] | 0.585 [0.577, 0.593] |

GIST は再利用済みの非独立 anchor で gate 非対象のため、SIFT と分けて全 2 operating points を記述する。

| GIST anchor | P method | beta (L2) | influenced queries n | F/P subset geomean [95% CI] | A/P subset geomean [95% CI] |
|---|---|---:|---:|---:|---:|
| gist-initial | native_P_beta0 | 0.000000 | 119 | 0.996 [0.993, 1.000] | 0.558 [0.551, 0.566] |
| gist-initial | native_P_factor_0p05 | 0.057683 | 119 | 0.997 [0.994, 1.002] | 0.559 [0.552, 0.568] |

A mismatch は `baseline_quality_mismatch_*_retained` と raw `baseline_validation_failure` に残した。

## 正の beta の記述的寄与

| P method | requested beta (L2) | session-query n | API p50/p95/p99 ms | beta=0 からの p50/p95/p99 変化 | median skipped groups | median vectors scanned (変化) |
|---|---:|---:|---:|---:|---:|---:|
| native_P_beta0 | 0.000000 | 200 | 8.516/9.289/9.723 | +0.000/+0.000/+0.000% | 20.0 | 7892.5 (+0.000%) |
| native_P_factor_0p01 | 2.481085 | 200 | 8.514/9.301/9.668 | -0.024/+0.130/-0.568% | 21.0 | 7796.0 (-1.223%) |
| native_P_factor_0p05 | 12.405425 | 200 | 8.371/9.244/9.679 | -1.695/-0.487/-0.462% | 24.5 | 7556.0 (-4.264%) |
| native_P_factor_0p1 | 24.810850 | 200 | 8.216/9.218/9.627 | -3.525/-0.767/-0.988% | 30.0 | 7139.0 (-9.547%) |

beta=0 から最大の保存済み正値 beta=24.810850 （session-query n=200）への比較では、正の beta が pruning と少なくとも一つの latency percentile に作用した。ただし同 operating point の A/P=0.597 [CI 0.591, 0.604] であり、この記述的寄与だけで gate 結論を変更しない。全 positive-beta P raw の nonzero observed gap は 0/7368 rows。主条件の positive-beta raw rows（repetitionを含む）n=1800 における max observed/certified/requested=0.000000/24.792015/24.810850 L2。observed gap は全て 0 であり、この実測 latency/pruning 差は観測された品質 gap との trade-off で得たものではない（他条件での gap 不発生は主張しない）。


## 主条件の timing component（p50 ms）

各値は各 session-query 内の repetition median を取った後の p50。`kernel/Δ-search` は native F/N/P では kernel total、A では Delta Flat search、`adaptive/merge` は native adaptive exact または A shortlist exact merge。`other micro` は micro からそれらの重複しない timer を引いた残差である。Base prepare と native query prepare は micro 外だが composed/API の層を明示するため併記する。

| role | method | beta | n session-query | Base prep | native prep | micro | kernel/Δ-search | LB | order | group scan | raw scan | adaptive/merge | receipt | other micro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F | native_F | — | 200 | 3.207 | 0.368 | 6.300 | 4.190 | 0.000 | 0.000 | 4.101 | 0.042 | 1.326 | 0.581 | 0.198 |
| N | native_N | — | 200 | 3.207 | 0.368 | 6.394 | 4.283 | 0.063 | 0.006 | 4.111 | 0.042 | 1.326 | 0.581 | 0.199 |
| A | faiss_A | — | 200 | 3.207 | 0.000 | 2.220 | 0.288 | — | — | — | — | 1.882 | — | 0.051 |
| P | native_P_beta0 | 0.000000 | 200 | 3.207 | 0.368 | 5.594 | 3.431 | 0.063 | 0.007 | 3.263 | 0.042 | 1.328 | 0.581 | 0.220 |
| P | native_P_factor_0p01 | 2.481085 | 200 | 3.207 | 0.368 | 5.547 | 3.401 | 0.063 | 0.007 | 3.232 | 0.042 | 1.328 | 0.581 | 0.224 |
| P | native_P_factor_0p05 | 12.405425 | 200 | 3.207 | 0.368 | 5.411 | 3.262 | 0.063 | 0.006 | 3.096 | 0.042 | 1.328 | 0.581 | 0.223 |
| P | native_P_factor_0p1 | 24.810850 | 200 | 3.207 | 0.368 | 5.248 | 3.090 | 0.063 | 0.006 | 2.925 | 0.042 | 1.330 | 0.581 | 0.222 |

`faiss_A_reference` は Faiss の squared-L2 順序を直接使う**非認証**の参考経路であり、RQ2 の A や gate には採用していない。その行は raw と代表 raw export に別 role `A-reference` で保存した。

## Fresh final

Validation は `NOT_PASSED` で終了した。ここでの研究 verdict は事前規定の validation stopping rule によるものであり、fresh-final holdout estimate ではない。candidate lock と final run config は作成せず、SIFT fresh query 1200..2199 と final-only HNSW reference は読み込んでいない。工学的導入判定は `NO_GO` である。

| final_decision field | value |
|---|---|
| final_status | `NOT_RUN_GATE_NOT_PASSED` |
| validation_gate_status | `NOT_PASSED` |
| performance_gate_status | `NOT_RUN_GATE_NOT_PASSED` |
| performance_verdict | `NOT_SUPPORTED_IN_TESTED_REGIME` |
| engineering_decision | `NO_GO` |
| selected_candidate | `null` |
| lock_created | `false` |
| large_final_started | `false` |
| fresh_sift_holdout_loaded | `false` |

## 品質・保証指標

match は同一 `C` の O との ordered key 一致率、recall は全 visible exact top-k に対する median/min、Delta capture は exact top-k に Delta neighbor がある queryだけの median/min（括弧内は該当 query-session 数）である。`n` は最初の repetition を代表とした query-session 数。P の gap は `max observed / max certified / max requested beta`（ordinary L2）で、必ずこの順の非減少 chain を満たすことを integrity gate が確認した。

| condition | method | n query-session | same-C match rate | full exact recall median/min | Delta capture median/min (n) | P gap obs/cert/req max |
|---|---|---:|---:|---:|---:|---:|
| gist-initial | faiss_A | 200 | 1.0000 | 1.0000/0.7000 | 1.0000/1.0000 (118) | — |
| gist-initial | faiss_A_reference | 200 | 1.0000 | 1.0000/0.7000 | 1.0000/1.0000 (118) | — |
| gist-initial | native_F | 200 | 1.0000 | 1.0000/0.7000 | 1.0000/1.0000 (118) | 0/0/0 |
| gist-initial | native_P_beta0 | 200 | 1.0000 | 1.0000/0.7000 | 1.0000/1.0000 (118) | 0/0/0 |
| gist-initial | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.7000 | 1.0000/1.0000 (118) | 0/0.0573129/0.0576829 |
| sift-delta-0 | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | — (0) | — |
| sift-delta-0 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | — (0) | — |
| sift-delta-0 | native_F | 200 | 1.0000 | 1.0000/0.9000 | — (0) | 0/0/0 |
| sift-delta-0 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | — (0) | 0/0/0 |
| sift-delta-0 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | — (0) | 0/0/12.4544 |
| sift-delta-1000 | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (16) | — |
| sift-delta-1000 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (16) | — |
| sift-delta-1000 | native_F | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (16) | 0/0/0 |
| sift-delta-1000 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (16) | 0/0/0 |
| sift-delta-1000 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (16) | 0/12.4456/12.4544 |
| sift-delta-100000 | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (197) | — |
| sift-delta-100000 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (197) | — |
| sift-delta-100000 | native_F | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (197) | 0/0/0 |
| sift-delta-100000 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (197) | 0/0/0 |
| sift-delta-100000 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (197) | 0/12.0841/12.0845 |
| sift-groups-512 | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-groups-512 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-groups-512 | native_F | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-groups-512 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-groups-512 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/12.4049/12.4054 |
| sift-groups-64 | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-groups-64 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-groups-64 | native_F | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-groups-64 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-groups-64 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/12.3862/12.4054 |
| sift-initial | faiss_A | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-initial | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | — |
| sift-initial | native_F | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-initial | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/0/0 |
| sift-initial | native_P_factor_0p01 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/2.46249/2.48108 |
| sift-initial | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/12.3379/12.4054 |
| sift-initial | native_P_factor_0p1 | 200 | 1.0000 | 1.0000/0.9000 | 1.0000/1.0000 (101) | 0/24.792/24.8108 |
| sift-k-1 | faiss_A | 200 | 1.0000 | 1.0000/0.0000 | 1.0000/1.0000 (12) | — |
| sift-k-1 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.0000 | 1.0000/1.0000 (12) | — |
| sift-k-1 | native_F | 200 | 1.0000 | 1.0000/0.0000 | 1.0000/1.0000 (12) | 0/0/0 |
| sift-k-1 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.0000 | 1.0000/1.0000 (12) | 0/0/0 |
| sift-k-1 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.0000 | 1.0000/1.0000 (12) | 0/10.4816/10.4895 |
| sift-k-100 | faiss_A | 200 | 1.0000 | 1.0000/0.9600 | 1.0000/1.0000 (200) | — |
| sift-k-100 | faiss_A_reference | 200 | 1.0000 | 1.0000/0.9600 | 1.0000/1.0000 (200) | — |
| sift-k-100 | native_F | 200 | 1.0000 | 1.0000/0.9600 | 1.0000/1.0000 (200) | 0/0/0 |
| sift-k-100 | native_P_beta0 | 200 | 1.0000 | 1.0000/0.9600 | 1.0000/1.0000 (200) | 0/0/0 |
| sift-k-100 | native_P_factor_0p05 | 200 | 1.0000 | 1.0000/0.9600 | 1.0000/1.0000 (200) | 0/13.8811/13.9003 |
| synthetic-clustered | faiss_A | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (57) | — |
| synthetic-clustered | faiss_A_reference | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (57) | — |
| synthetic-clustered | native_F | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (57) | 0/0/0 |
| synthetic-clustered | native_P_beta0 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (57) | 0/0/0 |
| synthetic-clustered | native_P_factor_0p05 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (57) | 0/0/0.155362 |
| synthetic-delta-near-queries | faiss_A | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (64) | — |
| synthetic-delta-near-queries | faiss_A_reference | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (64) | — |
| synthetic-delta-near-queries | native_F | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (64) | 0/0/0 |
| synthetic-delta-near-queries | native_P_beta0 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (64) | 0/0/0 |
| synthetic-delta-near-queries | native_P_factor_0p05 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (64) | 0/0/0.187547 |
| synthetic-high-dimensional-isotropic | faiss_A | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (53) | — |
| synthetic-high-dimensional-isotropic | faiss_A_reference | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (53) | — |
| synthetic-high-dimensional-isotropic | native_F | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (53) | 0/0/0 |
| synthetic-high-dimensional-isotropic | native_P_beta0 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (53) | 0/0/0 |
| synthetic-high-dimensional-isotropic | native_P_factor_0p05 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (53) | 0/0/0.703947 |
| synthetic-outlier-radius | faiss_A | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (61) | — |
| synthetic-outlier-radius | faiss_A_reference | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (61) | — |
| synthetic-outlier-radius | native_F | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (61) | 0/0/0 |
| synthetic-outlier-radius | native_P_beta0 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (61) | 0/0/0 |
| synthetic-outlier-radius | native_P_factor_0p05 | 64 | 1.0000 | 1.0000/1.0000 | 1.0000/1.0000 (61) | 0/0/0.0897634 |

## Geometry audit

全 13 audit files は、condition ごとの固定 validation/calibration subset（件数は表示）を使い、合計 condition-query sets 325 / 35125 group decisions（scan 25248、skip 9877）を timing 外で照合した。主 `sift-initial` は audit 25 queries / 3200 decisions であり、timing の別 test partition 200 queries 全件を audit したものではない。audit scope は `validation_queries_only_outside_timing`。保存した厳密 group 最小距離と native `LB` を照合し、`exact_min_lower - LB` を下界の緩さとして scan/skip 別に記述する。全 audit は completion の ancillary SHA-256 inventory と build の dataset/split ID に一致し、`passed=true` のものだけを集計した。

| condition | audited queries | action | decisions | radius median | true group-min L2 median | (true min-LB) median/p95 |
|---|---:|---|---:|---:|---:|---:|
| gist-initial | 25 | scan | 3065 | 1.69302 | 1.6822 | 1.57192/2.51566 |
| gist-initial | 25 | skip | 110 | 1.71062 | 2.54835 | 1.67545/2.55552 |
| sift-delta-0 | 25 | none | 0 | — | — | — |
| sift-delta-1000 | 25 | scan | 1964 | 286.143 | 402.834 | 320.349/385.981 |
| sift-delta-1000 | 25 | skip | 1236 | 275.289 | 605.376 | 298.23/349.46 |
| sift-delta-100000 | 25 | scan | 2652 | 368.092 | 376.156 | 312.064/358.716 |
| sift-delta-100000 | 25 | skip | 548 | 345.807 | 560.625 | 302.806/340.576 |
| sift-groups-512 | 25 | scan | 7928 | 297.835 | 392.823 | 309.109/370.204 |
| sift-groups-512 | 25 | skip | 4872 | 284.284 | 596.519 | 290.927/346.111 |
| sift-groups-64 | 25 | scan | 1344 | 363.993 | 395.662 | 334.409/382.752 |
| sift-groups-64 | 25 | skip | 256 | 341.63 | 583.01 | 326.613/368.76 |
| sift-initial | 25 | scan | 2445 | 344.279 | 390.451 | 325.541/381.721 |
| sift-initial | 25 | skip | 755 | 313.249 | 586.516 | 304.316/360.617 |
| sift-k-1 | 25 | scan | 2231 | 344.279 | 375.36 | 324.863/380.208 |
| sift-k-1 | 25 | skip | 969 | 315.932 | 581.4 | 309.997/369.624 |
| sift-k-100 | 25 | scan | 2655 | 342.676 | 403.689 | 324.865/381.376 |
| sift-k-100 | 25 | skip | 545 | 309.784 | 593.228 | 297.751/351.2 |
| synthetic-clustered | 25 | scan | 48 | 3.20917 | 2.92623 | 2.86479/3.52201 |
| synthetic-clustered | 25 | skip | 327 | 3.34855 | 44.7044 | 2.58002/3.40675 |
| synthetic-delta-near-queries | 25 | scan | 400 | 4.21452 | 4.67594 | 4.10285/5.31377 |
| synthetic-high-dimensional-isotropic | 25 | scan | 400 | 12.1718 | 15.0395 | 14.7259/15.9218 |
| synthetic-outlier-radius | 25 | scan | 116 | 222.062 | 19.8269 | 19.8269/32.249 |
| synthetic-outlier-radius | 25 | skip | 259 | 1.85296 | 25.7315 | 1.42917/7.12399 |

## Build / memory と限定的 break-even

主条件の cold/warm construction、cache audit、保持 memory は以下。RSS 差はプロセス全体の high-water/allocator 効果を含み、P 固有 memory とはみなさない。全 query timing は保存済みの warm immutable packed view で、各 condition の timing 前 warmup query 後に実行した。

| object | phase/metric | value | denominator / scope |
|---|---|---:|---|
| group centers | cold build | 2662.176 ms | build n=1; Base-only training |
| group assignment | cold build | 12.824 ms | build n=1 |
| group packing/radius | cold build | 53.654 ms | build n=1 |
| group total | cold build | 2728.653 ms | build n=1 |
| native packed view | cold build | 50.201 ms | build n=1 |
| native packed view | warm cache identity lookup | 0.002004 ms | build n=1 |
| Base visible map | cold cache build | 35.272 ms | audit files n=1; misses median=1 |
| Base visible map | cached / legacy p50 | 2.704/38.065 ms (14.050x) | paired audit queries n=25; hits median=344; total lookup median=210388 ns |
| groups | retained memory | 5136384 bytes | member + metadata; build n=1 |
| packed view | retained memory | 11257232 bytes | Python + native owned; build n=1 |
| groups + packed view | retained memory scenario | 16393616 bytes | additive sensitivity scenario; build n=1 |
| process | RSS change during build | 280756224 bytes | includes shared Base/Delta/truth/benchmark buffers; not P-only attribution |

`Q_break_even = B_extra / (t_F - t_P)` とし、分母は query-paired API wall 差の中央値で近似した。主列の `B_group` は center training + assignment + group packing/radius だけで、native packed view を F/P 共通の warm prerequisite と扱う。`感度 Q(group+packed)` は packed cold build も全て P に課した場合であり、両者の間に実システムの分担があり得る。更新頻度、永続化 I/O、共通 layout の利用範囲を含まない感度分析であり、`t_F - t_P <= 0` なら有限解なしとする。

| condition | P method | B_group ms | packed cold ms | median (t_F-t_P) ms/query | Q(group-only) | 感度 Q(group+packed) | n paired session-query | group bytes | group+packed bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gist-initial | native_P_beta0 | 9482.744 | 71.303 | -0.196408 | 有限解なし | 有限解なし | 200 | 19501568 | 59323040 |
| gist-initial | native_P_factor_0p05 | 9482.744 | 71.303 | -0.191999 | 有限解なし | 有限解なし | 200 | 19501568 | 59323040 |
| sift-delta-0 | native_P_beta0 | 2739.286 | 0.307 | 0.007809 | 350785.7 | 350825.1 | 200 | 67584 | 67600 |
| sift-delta-0 | native_P_factor_0p05 | 2739.286 | 0.307 | 0.005078 | 539495.0 | 539555.5 | 200 | 67584 | 67600 |
| sift-delta-1000 | native_P_beta0 | 2789.435 | 5.250 | 0.033954 | 82152.2 | 82306.8 | 200 | 574464 | 1823696 |
| sift-delta-1000 | native_P_factor_0p05 | 2789.435 | 5.250 | 0.056650 | 49239.4 | 49332.1 | 200 | 574464 | 1823696 |
| sift-delta-100000 | native_P_beta0 | 3224.890 | 692.425 | 3.111722 | 1036.4 | 1258.9 | 200 | 50755584 | 162092816 |
| sift-delta-100000 | native_P_factor_0p05 | 3224.890 | 692.425 | 4.505475 | 715.8 | 869.5 | 200 | 50755584 | 162092816 |
| sift-groups-512 | native_P_beta0 | 9610.054 | 48.461 | 1.164205 | 8254.6 | 8296.2 | 200 | 5339136 | 17008016 |
| sift-groups-512 | native_P_factor_0p05 | 9610.054 | 48.461 | 1.281796 | 7497.3 | 7535.1 | 200 | 5339136 | 17008016 |
| sift-groups-64 | native_P_beta0 | 1572.937 | 46.787 | 0.338334 | 4649.1 | 4787.4 | 200 | 5102592 | 16291216 |
| sift-groups-64 | native_P_factor_0p05 | 1572.937 | 46.787 | 0.506788 | 3103.7 | 3196.1 | 200 | 5102592 | 16291216 |
| sift-initial | native_P_beta0 | 2728.653 | 50.201 | 0.776782 | 3512.8 | 3577.4 | 200 | 5136384 | 16393616 |
| sift-initial | native_P_factor_0p01 | 2728.653 | 50.201 | 0.777665 | 3508.8 | 3573.3 | 200 | 5136384 | 16393616 |
| sift-initial | native_P_factor_0p05 | 2728.653 | 50.201 | 0.902544 | 3023.3 | 3078.9 | 200 | 5136384 | 16393616 |
| sift-initial | native_P_factor_0p1 | 2728.653 | 50.201 | 1.062119 | 2569.1 | 2616.3 | 200 | 5136384 | 16393616 |
| sift-k-1 | native_P_beta0 | 2745.743 | 46.180 | 1.145487 | 2397.0 | 2437.3 | 200 | 5136384 | 16393616 |
| sift-k-1 | native_P_factor_0p05 | 2745.743 | 46.180 | 1.283783 | 2138.8 | 2174.8 | 200 | 5136384 | 16393616 |
| sift-k-100 | native_P_beta0 | 2744.163 | 47.184 | 0.395431 | 6939.7 | 7059.0 | 200 | 5136384 | 16393616 |
| sift-k-100 | native_P_factor_0p05 | 2744.163 | 47.184 | 0.513875 | 5340.1 | 5432.0 | 200 | 5136384 | 16393616 |
| synthetic-clustered | native_P_beta0 | 4.293 | 0.742 | -0.015114 | 有限解なし | 有限解なし | 64 | 9344 | 39648 |
| synthetic-clustered | native_P_factor_0p05 | 4.293 | 0.742 | -0.010544 | 有限解なし | 有限解なし | 64 | 9344 | 39648 |
| synthetic-delta-near-queries | native_P_beta0 | 4.409 | 0.725 | -0.001673 | 有限解なし | 有限解なし | 64 | 9344 | 39824 |
| synthetic-delta-near-queries | native_P_factor_0p05 | 4.409 | 0.725 | -0.012667 | 有限解なし | 有限解なし | 64 | 9344 | 39824 |
| synthetic-high-dimensional-isotropic | native_P_beta0 | 5.440 | 0.857 | -0.018571 | 有限解なし | 有限解なし | 64 | 72960 | 232464 |
| synthetic-high-dimensional-isotropic | native_P_factor_0p05 | 5.440 | 0.857 | -0.016514 | 有限解なし | 有限解なし | 64 | 72960 | 232464 |
| synthetic-outlier-radius | native_P_beta0 | 4.257 | 0.715 | -0.009366 | 有限解なし | 有限解なし | 64 | 9344 | 39648 |
| synthetic-outlier-radius | native_P_factor_0p05 | 4.257 | 0.715 | -0.009234 | 有限解なし | 有限解なし | 64 | 9344 | 39648 |

## 段階 ablation と原因分解

以下は保存された実測値である。paired 行の `before/after ratio` は query-paired geometric mean であり、p50 列同士の単純比とは限らない。1 より大きいほど after が速い。旧 P 比較だけは query 集合が異なるため、表示 p50 の比を記述するだけで paired 推論には使わない。

| stage | metric | before | after | before/after ratio | scope / quality |
|---|---|---:|---:|---:|---|
| P rescan → heap | micro p50 ms | 8.535 | 5.594 | 1.516 | 200 paired session-query |
| F all-exact → adaptive | micro p50 ms | 17.175 | 6.300 | 2.710 | 200 paired session-query |
| Base legacy rebuild → cached | audit wall p50 ms | 38.065 | 2.704 | 14.050 | 25 paired audit queries; candidate hash/key identical |
| old P → packed/native P | micro p50 ms | 74.589 | 5.594 | 13.333 | **nonpaired** development n=5 vs validation session-query n=200; old E2E/new API=84.839/8.516 ms |
| A-reference → A | API p50 ms | 3.152 | 4.866 | 0.644 | 200 paired session-query; recall med/min 1.0000/0.9000 → 1.0000/0.9000; exact rechecks 0（非認証 float 順序、exact recheck なし） → 1.0 |

1. **旧 layout → packed native**: 旧 Python P-only development profile は保存済みであり、cProfile 自体の膨張値を性能推定には使わない。対応する unprofiled evidence の抜粋は `{"completed_blocks": 1, "completed_sha256": "0d8f2802e78678d1f2576ef106789d2c6a74d550fd71f11ad5875b01d076eb4d", "delta_flat_e2e_milliseconds": 23.565789, "delta_flat_micro_milliseconds": 13.226849, "median_definition": "middle of five sorted per-query values; descriptive development statistic", "old_pruning_beta0_component_medians_milliseconds": {"candidate_raw_distance": 0.477375, "group_ordering": 1.100606, "group_scan": 12.045408, "lb_calculation": 10.470187, "merge": 19.460495, "receipt": 0.798468, "visibility": 1.398842}, "old_pruning_beta0_e2e_milliseconds": 84.838983, "old_pruning_beta0_mean_exact_rechecks": 10.0, "old_pruning_beta0_mean_groups_skipped": 23.2, "old_pruning_beta0_mean_vectors_read": 7714.6, "old_pruning_beta0_micro_milliseconds": 74.589289, "raw_sha256": "59a8754b73a74bc24f54df2eba56221a18a24b8821b5335d61c64c2d5c517fa3", "run_id": "native-recheck-old-python-profile-20260905T145629.346102Z-10eec0e992", "total_blocks": 1}`。
2. **rescan → incremental heap**: `ablation_P_beta0_rescan_adaptive` と主 `native_P_beta0` は同じ P・beta=0・adaptive ranking を使い、各group decisionでの kth threshold 更新方式だけを変えた。F の rescan/heap は最終順位計算寄与の補助値に限る。
3. **all-boundary exact → adaptive exact**: `ablation_F_heap_all_exact` と主 F は heap を共通にし、境界 exact 範囲だけを変えた。
4. **no pruning → pruning**: N/P の paired比で group pruning の純寄与を分離した。
5. **Base visible-map rebuild → immutable cache**: validation queryだけの非計時 auditで両者の candidate hash/key が完全一致することを確認し、wall time と hit/miss/rebuild counterを各 runの `audits/*.base-cache.json` に保存した。

## 測定範囲と限界

- CPU 1 thread、immutable snapshot、Base HNSW `M=32, efConstruction=200, efSearch=128`、既定 `k=10, C=64, groups=128`。center は Base のみで学習し、beta は measurement query ではなく validation query の kth L2 medianから固定した。
- SIFT は Base 100k / Delta 10kを主条件、Delta 0/1k/100k、groups 64/128/512、k 1/10/100を検証した。GIST は memory 制約どおり Base 50k / Delta 5k / d=960。4 synthetic familyも実行した。
- validation は SIFT query 0..399 の再利用領域だけを読み、事前登録 fresh final holdout 1200..2199 は gate 通過前に読み込んでいない。GIST 全 query は過去使用済みなので独立 holdoutとは呼ばない。
- API wall は Base candidate preparationからReceipt完成までを全 method の独立randomized passで測り、その後に別順序のmicro passを実行した。composed E2E は共通 Base preparation + method固有 preparation + micro、micro は frozen C/native query preparation後の核である。build、oracle、audit、profileは query timing外。
- Formal validation は 1 process session のみで、process/session 間分散は validation CI に含まれない。Issue #3 で許された optional post-validation tuning は 0/2 rounds で、measurement partition を使った center/group/beta の再調整は行っていない。
- shared hostでexclusive CPU reservationはない。静的SIFT/GISTは更新時系列、text embedding、production DBMSを代表しない。

## Evidence identity

| run | config SHA-256 | implementation-tree SHA-256 |
|---|---|---|
| `native-recheck-validation-20260905T191058023915Z-227a918c57` | `227a918c5741f10ac138ce45217ac583dc118f044679ab44e64587c3b8da8925` | `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f` |

current implementation-tree / native shared-object SHA-256 はそれぞれ `bb9c65d7060ed8d8dfca3ff157b32aa318181f0a0b793eb816d12987fdd82c4f` / `eef245f3d812b3e90466ceee8ea0107ee72de72ed81fd706304e390042cd100a` で、correctness・validation、および final が lock 済みなら lock と一致することを report 生成時にも再検証した。

大容量 raw は report 入力 root `results/native_recheck_runs/validation` の各 run の `raw/` にあり、git 管理対象外である。下表の inventory digest は `(relative path, SHA-256, rows, bytes)` の canonical JSON に対する SHA-256、completion は ancillary/build/audit を束縛する marker の SHA-256 である。各 shard の相対 path・SHA-256・row 数の完全 inventory は hash-bound analysis JSON の `runs[].raw_files` にある。

| run | raw location (input-root relative) | shards | rows | bytes | raw inventory SHA-256 | completion SHA-256 |
|---|---|---:|---:|---:|---|---|
| `native-recheck-validation-20260905T191058023915Z-227a918c57` | `native-recheck-validation-20260905T191058023915Z-227a918c57/raw` | 84 | 42064 | 288076713 | `d31218eb63a96136c1e553430f5d10f5ec8b8f7b5aa8a07e3e45762ed2049240` | `10665158cdad322c0cf37926d6e21f8225e61ca705972ddfda97c4491bce5366` |

保存 raw がある環境では `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 make native-report NATIVE_VALIDATION_INPUT="results/native_recheck_runs/validation"` で checksum 検証から report を再生成できる。実験自体の再実行は同じ 1-thread 環境で `make native-validate`。raw JSONL は非圧縮であり、別の圧縮 artifact は主張しない。

追跡対象の summary/gate、deterministic representative sample、correctness、final decision、policy/holdout registration を合わせれば、研究 verdict、run/completion identity、全 raw shard checksum inventory、代表 row は第三者検査できる。summary/gate/representative の三点だけで確認できるのは validation outcome までである。ただし gitignore 済み raw/run directory がなければ全 query 行の再集計、ancillary audit の再読込、latency 分布の独立再計算はできず、summary から raw evidence を復元することもできない。

- analysis SHA-256: `1735992cd55f13bbe5281e354bba8af8bccaf10f073e04f7941e6569e788f41c`
- gate SHA-256: `17e04e5bfbd592dcfee4b43604fdd6bb7860aafa6e29501dfaf2c191a02f5b92`
- correctness SHA-256: `dd2cd9b1e50b4202e2d39e39acc36b48a317fdc8908acff6630bc3a56c0e22d5`
- old profile summary SHA-256: `5c283745294d8674b1d2f90e7fedbd68c0bfa7143412b6d0feaa80b61c6ef185`
- deterministic representative raw SHA-256: `bd831a851d52564cad074073076c3adcd4c72da2412a9cc568558989a5a993a7`
- final decision SHA-256: `7abd3aee91670bfaf0dffcf344597536812072b78151828aa28678af04c91029`
- final summary/gate SHA-256: `not run` / `not run`
- incomplete run は analysis から除外され、一覧は summary の `excluded_incomplete_runs` に残る。

この結果は指定条件の検索方式比較であり、production DBMS 全体の優位性、ACID commit throughput、一般の embedding 分布への外挿を主張しない。
