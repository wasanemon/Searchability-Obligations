# Searchability Obligations / Impact-Bounded Freshness 検証報告

- 報告基準日: 2026-09-05（Asia/Tokyo）
- 対象: [GitHub Issue #1](https://github.com/wasanemon/Searchability-Obligations/issues/1)
- 総合判定: **`NOT_SUPPORTED_IN_TESTED_REGIME`**
- 主実験の検索実装 SHA-256: `d56668c2585afac1cbc15e99720d41bedb5967ead6ff25288321ee7c26aa1c76`
- 最終集計証拠 SHA-256: `ac6eb0cf49371588690aec15ea416dd7421e665525bb036f4949081e08d1e72c`

## 0. 要旨

immutable な Base HNSW にまだ反映されていない commit 済み vector を Delta として保持し、
固定 center と covering radius の下界で Delta group を省略する方式を実装・検証した。保証対象は
DB 全体の exact top-k ではなく、一度だけ凍結した同じ Base 候補集合 `C` に visible Delta を
全走査した結果に対する、k 番目の**通常 L2 距離**の追加悪化である。

結論は negative result である。

| 仮説 | 判定 | 要点 |
|---|---|---|
| H1 / correctness | `SUPPORTED_IN_TESTED_REGIME` | final 112,800 生行で契約違反 0、同一 `C` の proposal recall 1.0、独立 Fraction oracle 16/16 一致、固定 seed 10,000 ケース通過 |
| H2 / usefulness | `NOT_SUPPORTED_IN_TESTED_REGIME` | pruning 本体 36 条件のすべてで optimized Delta Flat より遅い。最良でも退化条件 Delta=0 の speedup 0.9727 `[0.9711, 0.9735]` |
| H3 / lifecycle | `SUPPORTED_IN_TESTED_REGIME` | SQLite transaction、MVCC、raw obligation、update/delete、process crash、generation publish/pin/recover/GC の参照テストが通過 |

SIFT 初期条件では Delta Flat の E2E p50 22.408 ms に対して `beta=0` pruning は
81.795 ms、paired speedup は 0.2780 `[0.2762, 0.2803]` だった。GIST 縮小条件でも
111.295 ms 対 176.882 ms、0.6294 `[0.6290, 0.6298]` である。SIFT Delta=100,000 では
27.621 ms 対 567.466 ms、0.05023 `[0.04913, 0.05073]` まで悪化した。group を省略する
計算量削減は確認できたが、Python 上の LB、group 分割走査、可視性、exact ordering、merge、
Receipt の費用を埋められなかった。

したがって、この実装のまま本格 DBMS 実装へ進むことは推奨しない。correctness/lifecycle artifact
は残す価値があるが、次へ進む条件は native packed kernel などで optimized Flat を上回る
validation-only の事前証拠が得られることとする。

## 1. 仮説、契約、非主張

### 1.1 検証した仮説

- H1: 同じ ANN 候補 `C` に visible Delta を全走査した reference に対し、group 省略由来の
  k 番目距離の追加悪化を query 指定 `beta` 以内にできる。
- H2: LB、group 管理、merge、exact boundary、Receipt まで含めても、optimized Delta full scan
  より速い非自明な領域が存在する。
- H3: update/delete、raw pending、snapshot、crash、generation 切替があっても、visible version の
  取りこぼしや stale version の返却を防げる。

### 1.2 契約

query `q`、MVCC snapshot `s`、pin した Base generation `g` を固定する。

```text
B_s,g       = g に入り、s で visible な version
Delta_s,g   = s で visible だが g に未反映の version（raw pending を含む）
C(q,s,g)    = visibility 処理・重複排除・不足時補充後に凍結した Base ANN 候補
R_ref       = TopK(C union Delta_s,g)
R_prop      = TopK(C union S), S subseteq Delta_s,g
0 <= tau_prop - tau_ref <= requested_beta
```

`C` は query/snapshot/generation、Base universe、version key、vector bits に bind し、reference と
proposal に同じ object を渡す。HNSW を二回検索して得た集合を「同じ `C`」とは扱わない。
`beta=0` では `(exact_squared_distance, logical_id, version_id)` の exact ordering まで一致させる。
`beta>0` の保証は k 番目距離だけであり、返却 ID、recall、全順位、経過時間 freshness は保証しない。

### 1.3 Group rule と数値契約

group `G` の固定 center `c` と全 member を覆う上向き radius `r_hi` に対して、

```text
LB(q,G) = max(0, lower(d(q,c)) - r_hi)
skip G iff LB(q,G) > upper(tau) - beta
```

を用いる。不等号は strict `>` で、`LB = tau-beta` は必ず scan する。raw pending は先に全走査し、
残りの group は `(LB, group_id)` 昇順に処理する。候補は追加だけなので `tau` は非増加であり、
一度行った skip 判定は後から無効にならない。省略 group の最小下界を `L_min` とすると Receipt は
`certified_beta=max(0,tau_exact_hi-L_min)` を返す。省略がなければ 0 である。

Faiss `METRIC_L2` の出力は squared-L2 だが、`radius/LB/tau/beta` と報告値は ordinary L2 である。
保証経路は canonical binary32 入力を binary64 `gamma_(d+2)` 区間で囲み、radius は上向き、LB は
下向き、tau は上向きにする。曖昧な最終境界は binary32 成分の exact `Fraction` で順位付けする。
epsilon-only や plain-float 経路を certified と呼んでいない。詳細は
[`docs/correctness.md`](../docs/correctness.md) と [`docs/numerics.md`](../docs/numerics.md) にある。

### 1.4 主張しないこと

- Base HNSW の miss を含む DB 全体 exact top-k 保証、最新再構築 HNSW との同一性。
- production DBMS、分散 transaction、高並列 MVCC、電源断・filesystem corruption 耐性。
- `beta>0` による有用な品質/速度 trade-off。今回は observed degradation がすべて 0 だった。
- 画像 descriptor の結果を text embedding、RAG、任意 workload へ一般化すること。
- 新規性、特許性、freedom-to-operate。先行研究上の最終判定は未確定である。

## 2. 実装範囲

- Python 3.10 / `faiss-cpu==1.15.0` の immutable `IndexHNSWFlat` Base。
- immutable `bytes` backing の vector/candidate/group payload、ordinal-to-version mapping。
- raw pending と immutable group、fixed center、conservative radius、連続 member matrix。
- visibility filtering、underfill 補充、frozen `C`、full Delta と certified pruning、exact tie ordering。
- 検索実装から独立した小規模 `Fraction` 全列挙 oracle。
- optimized Delta Flat、Delta HNSW、全体 HNSW、全体 exact Flat、no-pruning、interval-recompute
  ablation を同じ harness で測る経路。
- SQLite の vector version と obligation の同一 transaction、snapshot、raw 再構成、group catalog、
  immutable generation の fsync/rename/publish/recover/pin と保守的 retention。
- block checkpoint、query-method-repetition JSONL、設定・dataset・split・実装 hash、構築・memory
  manifest、atomic completion receipt、再集計器。

候補不足、世代不在、壊れた bound、HNSW underfill は、Base 全列挙補充、該当 group scan、全 visible
exact、または適用不能表示へ安全側に倒す。NaN/Inf、complex、次元不一致、範囲外、負の beta、
不正 k は certified result の前に拒否する。

## 3. データと事前固定した設定

### 3.1 Provenance

| データ | 用途 | 次元 | 取得・検証 |
|---|---|---:|---|
| synthetic clustered | 分離 cluster | config 固定 | current smoke 完走 |
| synthetic isotropic | 高次元・弱い bound | config 固定 | current smoke 完走 |
| synthetic delta-near-queries | query に近い更新 stress | config 固定 | current smoke 完走 |
| synthetic outlier-radius | radius 膨張 stress | config 固定 | current smoke 完走 |
| TEXMEX SIFT1M | 主比較・sweep | 128 | archive 168,280,445 bytes、official MD5 一致、SHA-256 `92f1270c...c81a` |
| TEXMEX GIST1M | 高次元縮小比較 | 960 | archive 2,740,172,684 bytes、official MD5 一致、SHA-256 `01469a7f...f73` |

公式 MD5、観測 SHA-256、展開各ファイルの SHA-256、CC0/source URL は
[`data/manifests/sift.json`](../data/manifests/sift.json) と
[`data/manifests/gist.json`](../data/manifests/gist.json) に保存した。独自 Base/Delta split なので、
配布 ground truth は流用せず、全 visible 集合に対する exact truth を再計算した。

### 3.2 Final matrix

| 条件 | Base / Delta | validation / test | repetition | k / C / groups | 備考 |
|---|---:|---:|---:|---|---|
| SIFT initial | 100,000 / 10,000 | 200 / 1,000 | 3 | 10 / 64 / 128 | Base `efSearch=128` |
| GIST initial reduced | 50,000 / 5,000 | 200 / 800 | 3 | 10 / 64 / 128 | memory 制約による縮小 |
| seed・一軸 sweep | 条件別 | 200 / 200 | 1 | 設定別 | test query で tuning しない |
| SIFT Delta heavy | 100,000 / 100,000 | 200 / 200 | 1 | 10 / 64 / 128 | 別 run で memory を制御 |

固定 sweep は Delta size `0/1,000/10,000/100,000`、groups `16/64/128/512`、`k=1/10/100`、
Base `efSearch=32/128/512`、group-build seed `0/1/2`、positive beta factor `0.01/0.05/0.10`、
comparator `efSearch=32/128/512` を含む。seed は data split seed ではなく k-means group-build seed である。
center は Base だけで学習し、beta の絶対値は validation の kth-distance median から決めた。

## 4. 測定と公平性

### 4.1 比較方式

| 方式 | 役割 |
|---|---|
| Base HNSW + optimized Delta Flat | 同じ `C` を使う主 reference。一回の Faiss full scan と exact boundary rerank |
| Base HNSW + pruning (`beta=0/>0`) | 提案方式 |
| Base HNSW + Delta HNSW | certificate のない実用比較 |
| Base+Delta 全体 HNSW | 反映済み ANN の recall-latency 比較 |
| Base+Delta exact Flat | 全 visible truth と別計時の exact 参考 |
| grouping, no pruning | group 管理・分割 scan overhead の ablation |
| final-interval recompute | interval carry 最適化の ablation |

公平性検査はすべて満たした。

- [x] coupled 方式は query ごとに同じ frozen `C` object を再利用した。
- [x] Delta Flat の計時へ correctness 用の二重 scan を混入させていない。
- [x] ordinary-L2、canonical dtype、dataset、k、Base index、候補数、1 thread を揃えた。
- [x] Base initial `efSearch=128` を固定し、Base-ef sweep は別条件にした。
- [x] frozen `C` 後の micro と Base 検索込み E2E を分離した。
- [x] truth、oracle、download、validation 選択、LB audit を query latency から除外した。
- [x] visibility、raw scan、LB、ordering、group scan、merge、exact boundary、Receipt は無料扱いしない。
- [x] 10 warmup query 後、method 順を query/repetition ごとに決定的に shuffle した。
- [x] HNSW は recall-latency 曲線で表示し、低品質設定だけを勝者にしない。
- [x] build/add/group/memory を別計測し、ACID commit throughput と呼ばない。
- [x] `COMPLETED.json`、checkpoint、manifest、effective config、全 raw shard の identity/hash/row count を再検証した。
- [x] pre-audit、incomplete、identity 不整合を主集計へ混ぜていない。

### 4.2 統計

主要条件の single-query percentile は 3 repetition を query 内で median 化してから、1,000 SIFT / 800
GIST query 上で p50/p95/p99 を取った。他の sweep は 200 query・1 repetition である。paired speedup は
同じ `(dataset hash, split ID, experiment, query_id)` の `t_delta_flat/t_method` とし、query を単位に
2,000 回の決定的 bootstrap を行った。seed は method ごとに最終 summary へ保存した。

実測 batch QPS は 25 query を 1 thread で逐次実行した wall timeから求めた。ただし各
method/condition につき 1 trial だけなので QPS の CI、run-to-run 分散、shared-host noise は未測定である。
`1/mean(single-query latency)` は summary で capacity estimate と明記し、実測 QPS と混同していない。
p99 は有限標本の empirical tail で、200/800 query 条件では特に粗い。

## 5. 証拠の完全性と正当性

### 5.1 採用 run

| run ID | config hash | block / raw row / bytes | completion checkpoint SHA-256 |
|---|---|---:|---|
| `texmex-main-and-lean-sweeps-20260905T085044.777697Z-98bb0214e0` | `98bb0214...a3909` | 176 / 110,400 / 598,994,358 | `8b7fa835...97075` |
| `texmex-sift-delta100k-heavy-sweep-20260905T113947.108257Z-e46edaa085` | `e46edaa0...f353` | 8 / 2,400 / 24,101,234 | `0a047147...8a07` |

両 run は同じ検索実装 hash `d56668c2...1c76` と atomic `COMPLETED.json` を持つ。集計器は completion
の run/config/implementation identity、checkpoint と run-manifest SHA-256、raw shard 数、status、
completed/total block 数、`object_sha256(effective_config)`、各 JSONL の SHA-256 と行数を fail-closed
で再照合した。最終集計は 184 shard、112,800 行、200 method-condition group である。

主 identity は次のとおりで、全 sweep の完全値は機械可読 summary にある。

| experiment | dataset hash | split ID |
|---|---|---|
| SIFT initial | `82ac0307...c97cba` | `455bb2ef...0cb2` |
| GIST reduced initial | `958b61ae...bd3ae` | `5775a602...83c` |
| SIFT Delta=100,000 | `6b408e07...e16fa` | `d4de9adf...4684` |

### 5.2 保存して除外した failure

最初の final attempt `texmex-main-and-lean-sweeps-20260905T062813.056342Z-98bb0214e0` は
104/176 block、約 460.6 MB を保存した後、empty Delta の ndarray hash で
`TypeError: memoryview: cannot cast view with zeros in shape or strides` となった。`COMPLETED.json` がなく、
状態は `failed_incomplete` のまま保存し、集計器は全 aggregate から除外した。zero-byte partition の
field name/dtype/full shape は hash し、payload だけを空として扱う修正と回帰テスト後に新 run を開始した。
旧 104 block と新実装の block は混在させていない。

### 5.3 実行済み correctness / recovery tests

| 検査 | 結果 |
|---|---|
| `make test` | 132 passed、1 deselected、8 warnings、7.44 s |
| `make test-full` | 133 passed、8 warnings、54.54 s |
| 固定 seed 10,000 randomized | 1 passed、46.94 s、seed `6202052` |
| final JUnit | SHA-256 `4701592b238fc7dff6bd30ae568dee41049723efcecf2ee1f26c3674b9e9bbd9` |
| final raw contract | violation 0、baseline validation failure 0、fallback 0 |
| 独立 Fraction oracle | 16 executed / 16 match / 0 skip |
| current synthetic smoke | 8/8 blocks、384 rows、violation 0、oracle 4/4 |

133 tests は empty/singleton/tie/radius=0、`beta>tau`、strict LB equality と両側 `nextafter`、
underfill、k 超過、duplicate/multiversion、visibility/delete、異なる scale、outlier、丸め境界、
NaN/Inf/complex、不正 input、radius coverage、immutable backing 改変反例、delete 後処理、center 移動、
早すぎる GC、小 beta でも ID が多数変わる反例を含む。`test_crash.py` 14 case と `test_store.py`
15 case は、実 process `os._exit(86)`、insert/update/delete obligation の transaction 境界、group と
generation の file/fsync/rename/manifest/catalog 境界、pin/recover/conservative GC を検査した。

warnings は NumPy が optional PyYAML の未導入を知らせるものだけで、隠さず出力に残した。

## 6. 主結果

### 6.1 SIFT initial

全距離は ordinary L2。speedup は `Delta Flat / method` なので 1 より大きいと提案方式が速い。

| method | E2E p50/p95/p99 ms | batch QPS | paired speedup `[95% CI]` | exact recall / same-C recall | group skip | vectors read |
|---|---:|---:|---:|---:|---:|---:|
| Delta Flat | 22.408 / 23.689 / 23.910 | 46.518 | 1.000 | 0.9989 / 1.0000 | N/A | 10,000 |
| pruning beta=0 | 81.795 / 89.966 / 91.421 | 12.207 | 0.2780 `[0.2762,0.2803]` | 0.9989 / 1.0000 | 21.924% | 7,488.7 |
| beta factor=.01 | 81.672 / 89.957 / 91.861 | 12.211 | 0.2784 `[0.2761,0.2815]` | 0.9989 / 1.0000 | 22.529% | 7,428.6 |
| beta factor=.05 | 80.183 / 89.610 / 91.106 | 12.374 | 0.2826 `[0.2809,0.2862]` | 0.9989 / 1.0000 | 24.992% | 7,186.3 |
| beta factor=.10 | 78.098 / 88.876 / 90.741 | 12.510 | 0.2905 `[0.2874,0.2936]` | 0.9989 / 1.0000 | 28.302% | 6,871.1 |

requested beta は factor `.01/.05/.10` で `2.481085 / 12.405425 / 24.810850`。certified beta の
p50/max は `0.211338/2.478150`、`10.200015/12.403919`、`22.791148/24.805021` で、すべて
`0 <= observed <= certified <= requested` を満たした。observed upper の最大は全条件 0 だった。

Delta が same-C reference top-k に入る query は 51.9%。Delta exact-neighbor capture は repetition を
query 内で畳んだ incidence で 909/909。Delta-influence subset の exact recall は 0.99904 で、全体の
0.9989 と同様、残差は主に frozen Base candidate 側である。

### 6.2 GIST reduced initial

| method | E2E p50/p95/p99 ms | batch QPS | paired speedup `[95% CI]` | exact recall / same-C recall | group skip | vectors read |
|---|---:|---:|---:|---:|---:|---:|
| Delta Flat | 111.295 / 113.330 / 114.717 | 9.103 | 1.000 | 0.987875 / 1.0000 | N/A | 5,000 |
| pruning beta=0 | 176.882 / 179.853 / 183.027 | 5.635 | 0.6294 `[0.6290,0.6298]` | 0.987875 / 1.0000 | 1.945% | 4,908.6 |
| beta factor=.01 | 177.268 / 180.143 / 181.928 | 5.651 | 0.6286 `[0.6282,0.6290]` | 0.987875 / 1.0000 | 1.994% | 4,906.9 |
| beta factor=.05 | 177.146 / 180.033 / 181.889 | 5.663 | 0.6285 `[0.6280,0.6289]` | 0.987875 / 1.0000 | 2.262% | 4,896.4 |
| beta factor=.10 | 176.971 / 179.952 / 181.551 | 5.666 | 0.6293 `[0.6290,0.6298]` | 0.987875 / 1.0000 | 2.612% | 4,881.6 |

requested beta は `.0115366/.0576829/.115366`、certified max は
`.0113544/.0573299/.115342`、observed upper max は 0。Delta influence は 58.0%、unique-query
Delta exact-neighbor capture は 707/707、subset exact recall は 0.98642 だった。GIST は 960 次元で
radius bound が弱く、beta を増やしても vector read はわずかしか減らなかった。

### 6.3 SIFT Delta=100,000

| method | E2E p50/p95/p99 ms | batch QPS | paired speedup `[95% CI]` | exact / same-C recall | skip / vectors |
|---|---:|---:|---:|---:|---:|
| Delta Flat | 27.621 / 32.761 / 38.116 | 38.328 | 1.000 | 0.999 / 1.000 | N/A / 100,000 |
| pruning beta=0 | 567.466 / 596.351 / 607.361 | 1.768 | 0.05023 `[0.04913,0.05073]` | 0.999 / 1.000 | 11.730% / 87,990 |
| beta factor=.05 | 555.082 / 594.487 / 609.219 | 1.791 | 0.05128 `[0.05010,0.05219]` | 0.999 / 1.000 | 14.520% / 85,266 |

Delta influence は 98.5%、capture は 953/953。positive beta requested `12.084539`、certified
p50/max `9.749569/12.084080`、observed max 0。Delta が大きいほど Flat の vectorized scan が有利で、
Python/group overhead は償却されなかった。

## 7. 補助 baseline の品質・速度

| dataset | method | E2E p50 ms | batch QPS | exact recall@k | 解釈 |
|---|---|---:|---:|---:|---|
| SIFT initial | Delta HNSW ef32 / 128 / 512 | 22.052 / 22.166 / 22.515 | 47.19 / 46.65 / 46.21 | .9985 / .9989 / .9989 | 同じ Base C、Delta 側は非保証 ANN |
| SIFT initial | full HNSW ef32 / 128 / 512 | .212 / .424 / 1.091 | 5964 / 2863 / 1058 | .9685 / .9989 / 1.000 | 全体を再反映した ANN、same-C 比較ではない |
| SIFT initial | full exact Flat performance | 16.344 | 63.244 | 1.000 | 110k 全体 scan の参考値 |
| GIST reduced | Delta HNSW ef32 / 128 / 512 | 110.004 / 110.311 / 110.915 | 9.274 / 9.193 / 9.207 | .98725 / .987875 / .987875 | ef32 は Delta miss あり |
| GIST reduced | full HNSW ef32 / 128 / 512 | .518 / 1.132 / 2.861 | 2453 / 1058 / 264 | .88475 / .985375 / .998 | recall-latency trade-off |
| GIST reduced | full exact Flat performance | 116.079 | 8.509 | 1.000 | 55k 全体 scan の参考値 |
| SIFT Delta=100k | Delta HNSW ef128 / 512 | 24.614 / 25.739 | 43.79 / 42.09 | .998 / .999 | Flat 比 1.123 / 1.080 倍だが deterministic certificate なし |

この規模では full HNSW が非常に速いが、build/add、更新 visibility、historical snapshot を同じものとして
比較していない。Delta HNSW は実用的な代替候補だが、本案の deterministic impact certificate を持たない。

## 8. Sweep、ablation、負ける条件

### 8.1 Delta size

| Delta | beta=0 p50 ms | Delta Flat p50 ms | paired speedup `[95% CI]` |
|---:|---:|---:|---:|
| 0 | 24.951 | 24.237 | 0.9727 `[0.9711,0.9735]` |
| 1,000 | 49.041 | 24.089 | 0.4923 `[0.4880,0.4957]` |
| 10,000 | 81.795 | 22.408 | 0.2780 `[0.2762,0.2803]` |
| 100,000 | 567.466 | 27.621 | 0.05023 `[0.04913,0.05073]` |

Delta=0 は実質的に fixed overhead だけを見る退化条件で、それでも CI 全体が 1 未満だった。

### 8.2 Group count

| groups | beta=.05 skip | pruning p50 ms | Delta Flat p50 ms | speedup |
|---:|---:|---:|---:|---:|
| 16 | 5.81% | 57.802 | 24.017 | .4150 |
| 64 | 13.88% | 70.670 | 24.716 | .3498 |
| 128 | 24.99% | 80.183 | 22.408 | .2826 |
| 512 | 34.53% | 186.354 | 24.743 | .1323 |

group を増やすと skip は増えたが、LB mean は 1.369→5.267→10.355→41.072 ms、ordering は
.073→.466→1.133→6.183 ms に増え、速度は単調に悪化した。これは「skip 率」を「処理削減」や
「高速化」と同一視できない例である。

### 8.3 k、Base ef、build seed

- `k=1/10/100` の beta=.05 speedup は `.1927/.2826/.6689`。大きい k では Flat merge も重くなり
  差は縮んだが、CI を含め 1 を超えない。
- Base `efSearch=32/128/512` の beta=0 speedup は `.2792/.2780/.2830`。exact recall は
  `.971/.9989/1.0` へ変わるが、same-C recall は常に 1.0 で契約と Base 品質を分離できた。
- group-build seed 0/1/2 の beta=0 p50 は SIFT `81.795/84.600/87.949` ms、GIST
  `176.882/181.469/178.799` ms。seed 0 は大きい主 query 集合、seed 1/2 は先頭 200 query なので、
  3-seed 数値は記述的感度であり母集団 CI ではない。それでも全 seed で Delta Flat より遅い。

### 8.4 Ablation

| 条件 | no-pruning p50 | beta=0 p50 | interval recompute p50 | 解釈 |
|---|---:|---:|---:|---|
| SIFT initial | 90.399 | 81.795 | 87.160 | pruning は分割 scan 比 8.604 ms 改善、interval carry は recompute 比 5.365 ms 改善 |
| GIST reduced | 177.270 | 176.882 | 211.637 | pruning の純改善は 0.388 ms、recompute は約 35 ms の費用 |
| SIFT Delta=100k | 594.354 | 567.466 | 632.773 | pruning は 26.888 ms 改善するが Flat との差は 539.845 ms |

改善版は各 ablation より良い場面がある。しかし optimized contiguous Delta Flat を上回るには
至らない。small-Delta full-scan fallback、packed multi-group native scan、outlier separation は
validation/profile で採択していないため未実装・未測定であり、実行済み改善として数えない。

## 9. LB、tau、radius と scan 理由

LB audit は主要条件の先頭 25 query に対する非計時 sample で、全 query 分布ではない。

| 条件 | LB p50/p95/p99 | radius p50/p95/p99 | tau p50/p95/p99 | audit 判定 |
|---|---:|---:|---:|---|
| SIFT initial | 144.021 / 319.584 / 361.471 | 334.399 / 380.819 / 393.211 | 250.591 / 302.815 / 324.725 | LB=0 scan 530、positive scan 2,073、strict skip 597 / 3,200 |
| GIST reduced | 0 / .9518 / 1.5285 | 1.6921 / 2.6369 / 3.0925 | 1.2152 / 1.7856 / 2.4204 | LB=0 scan 1,932、positive scan 1,153、strict skip 90 / 3,175 |
| SIFT Delta=100k | 121.734 / 275.865 / 320.673 | 365.085 / 399.371 / 409.408 | 251.951 / 296.547 / 307.771 | zero 26.66%、positive scan 62.56%、skip 10.78% |

SIFT heavy では radius median が tau median より大きく、多くの下界が弱い。GIST は 60.85% の
audit 判定が `LB=0` である。実データ audit では equality は 0 件だったため、strict equality は
専用の exact/`nextafter` test が担う。全 final method group の fallback は 0 で reason histogram は空。

## 10. 費用内訳、構築、memory

### 10.1 Query component

以下は beta=0 の raw request あたり mean。`base_search` は candidate preparation wall の内部成分で、
残りと単純加算できない。Python materialization と未区分 residual もある。

| 条件 | Base prepare wall | visibility | raw/candidate | LB | ordering | group scan | merge | Receipt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT initial | 9.623 ms | 1.363 | .482 | 10.353 | 1.132 | 11.213 | 19.039 | 1.059 |
| GIST reduced | 8.450 ms | .866 | .861 | 11.218 | .887 | 25.544 | 109.190 | .894 |

Delta Flat の別成分は SIFT で Delta scan .447 ms + merge 12.627 ms、GIST で 1.592 + 101.979 ms。
proposal は vector read を SIFT 25.1%、GIST 1.83% 減らしたが、LB と分割 group scan の追加費用が
それを超えた。Faiss API から信頼できる per-query HNSW visited count は取得できず `N/A` とした。

### 10.2 Build / maintenance component

| 項目 | SIFT initial | GIST reduced | SIFT Delta=100k |
|---|---:|---:|---:|
| Base HNSW build | 26.375 s | 41.476 s | 26.194 s |
| Delta HNSW add | 1.241 s | 1.506 s | 27.779 s |
| full HNSW add | 29.667 s | 46.365 s | 69.077 s |
| center training | 2.6766 s | 9.4146 s | 2.6605 s |
| Delta assignment | .0130 s / 9,900 vectors | .0447 s / 4,950 | .1278 s / 99,000 |
| packing + radius | .0542 s | .0813 s | .3493 s |
| visible-view cache | .0082 s | .0106 s | .0751 s |
| group pipeline total | 2.7520 s | 9.5512 s | 3.2127 s |
| group extra vs Delta Flat construct/add | 2.7490 s | 9.5417 s | 3.1836 s |

これらは逐次 component build cost であり、SQLite ACID commit latency、write throughput、同時 query
下の backlog 安定性ではない。1-vector assignment latency と batch-size 感度も未測定である。

### 10.3 Memory

| 条件 | RSS 増分 / process high-water | Base + Delta ndarray | group member + metadata | serialized comparison indexes 合計 |
|---|---:|---:|---:|---:|
| SIFT initial | 428.3 / 786.4 MiB | 48.8 + 4.9 MiB | 4.83 + .06 MiB | 223.1 MiB |
| GIST reduced | 1,258.8 / 4,836.1 MiB | 183.1 + 18.3 MiB | 18.13 + .47 MiB | 651.1 MiB |
| SIFT Delta=100k | 918.7 / 1,536.4 MiB | 48.8 + 48.8 MiB | 48.34 + .06 MiB | 445.6 MiB |

RSS/high-water は各 comparator index、allocator retention、build temporary の共存を含み、proposal 単独の
resident memory ではない。group 固有 payload は表の member/metadata だが、Python object overhead や
index 内部 allocator を完全には分離していない。

## 11. Break-even

query-side の差を `Delta_t=t_flat-t_prop` とすると、SIFT initial beta=0 は `-59.387 ms`、GIST は
`-65.587 ms`、heavy は `-539.845 ms`。pruning 本体 36 条件と recompute 16 条件のどれにも
paired CI 下端が 1 を超える条件はなく、tested grid 内に crossing はない。未測定点を補間・外挿して
break-even を作っていない。

maintenance の追加 build costを `B_extra`、query 数を `Q` とする単純式
`Q_break_even=B_extra/(t_flat-t_prop)` でも、全測定条件で分母が 0 以下なので有限の正の解はない。
更新維持費を無視しても query が遅いため、amortization で結論は反転しない。

## 12. Lifecycle safety

version row と searchability obligation は同じ SQLite transaction sequence へ書き、commit 後に raw
pending として復元可能にした。query は snapshot と generation/group revision を pin し、visibility を
検索前に適用する。group publish は catalog revision と content binding を持ち、immutable generation は
一時 prefix、file fsync、rename、directory fsync、manifest、catalog publish の順で公開する。reader pin が
ある世代は GC しない。保持世代より古い snapshot は retained exact fallback、または明示拒否にする。

実 process crash tests は transaction 内、commit 直後、update/delete interval と obligation の間、
generation file/rename/manifest/catalog、group publish 境界を注入した。再起動後の許容状態は「旧世代 +
raw obligation」または「完全に有効な新世代」で、visible vector が neither になる状態を許さない。

この支持範囲は小規模参照 store の tested fault points に限る。`PRAGMA synchronous` を含む実 hardware
power loss、filesystem corruption、distributed commit、高並列 race、長期 retention/storage growth は
未測定で、production safety の主張ではない。

## 13. 先行研究と新規性

一次資料との詳細差分は [`docs/related_work.md`](../docs/related_work.md) に保存した。

- Base graph と未反映 journal/delta の併用は Oracle 型 journal、FreshDiskANN などで既知。
- center/radius と triangle inequality による strict branch-and-bound は M-tree/ball-tree 系で既知。
- kth distance の additive error、query-specific certificate も一般概念として先行例がある。
- HNSW の empirical recall、理論 ANN の乗法誤差、version/time bounded staleness は本契約と別物。

残る可能性がある差分は、durable obligation/snapshot/generation と、同じ frozen Base candidate `C` に
対する Delta 省略だけの query-specific additive certificate を一体で契約化した点である。ただし同じ
組合せが論文・製品・特許にないことは立証していない。新規性の結論は `INCONCLUSIVE` であり、
「初の journal-aware ANN」「初の additive stream k-NN」「新しい center-radius pruning」とは主張しない。

## 14. 妥当性への脅威と未測定事項

- SIFT/GIST は静的画像 descriptor で、現実の更新時系列、text embedding、RAG を代表しない。
- GIST は Base 50k / Delta 5k / test 800 の縮小条件。SIFT と同規模ではない。
- Base HNSW の誤差は契約外。exact recall は別測定し、`beta` へ合算していない。
- 主要条件以外は 200 query・1 repetition。batch QPS は 1 trial、共有 host で exclusive CPU ではない。
- center/group/beta の最終 query tuning はしていないが、tested matrix 自体は限定的である。
- sustained mixed workload、write throughput、backlog、再構築頻度、assignment batch-size、energy は未測定。
- ACID commit と ANN add/build の同期 latency、HNSW visited count、方式単独 RSS は未分離。
- power loss、filesystem/hardware corruption、長期 GC/storage、high-concurrency snapshot は未測定。
- small-Delta fallback、packed/native multi-group scan、outlier separation は未実装。
- 先行研究は網羅的 systematic review や patent/FTO opinion ではない。
- positive beta でも observed gap は 0 で、非ゼロ品質損失との trade-off を実証していない。
- 一度 failed になった empty-Delta run は原因と raw を残したが、主集計から正しく除外した。

## 15. 最終判断と次段階

| 仮説 | 最終評価 | 根拠 | 限定 |
|---|---|---|---|
| H1 | 支持 | 112,800 raw rowsで violation 0、same-C recall 1、observed<=certified<=requested、oracle 16/16、10k randomized | finite tested input と frozen-C comparator のみ |
| H2 | 不支持 | 全 pruning 条件の speedup CI が 1 未満、query/build break-even なし | 現 Python/Faiss/group kernel と tested grid のみ |
| H3 | 支持 | store/crash 29 caseを含む統合 suite 通過 | reference lifecycle と injected process faults のみ |

よって総合 enum は一つだけ、**`NOT_SUPPORTED_IN_TESTED_REGIME`** とする。correctness と lifecycle が
成立しても、Issue #1 が要求する実用上の非自明な勝ち領域は見つからなかったためである。

次段階を行うなら、最初に validation-only で packed native group scan、vectorized LB、contiguous member
layout を profile し、Delta Flat に対する E2E paired CI 下端が 1 を超えたときだけ real final query へ
進む。続いて mixed read/write、retention、native DB integration を測る。現状の Python prototype をそのまま
本格 DBMS へ拡張する判断はしない。

## 16. 再現手順と証拠

標準手順は次である。

```bash
make setup
make test
make test-full
make smoke
make data
make evaluate
make report
```

保存場所:

- 大容量 per-query raw: `results/runs/<run-id>/raw/*.jsonl`（Git 外、ローカル保存）
- completion/checkpoint/build/split/validation: 各 run directory
- 小容量 immutable final snapshot: [`results/final_evidence/summary.json`](../results/final_evidence/summary.json)
- snapshot manifest: [`results/final_evidence/manifest.json`](../results/final_evidence/manifest.json)
- fixed-seed JUnit: [`results/test_evidence/randomized_10000_final.xml`](../results/test_evidence/randomized_10000_final.xml)
- 自動集計素材: [`REPORT_ja.generated.md`](REPORT_ja.generated.md)
- 実験図: [`figures/latency_quantiles.png`](figures/latency_quantiles.png)、
  [`figures/quality_latency.png`](figures/quality_latency.png)、
  [`figures/group_scan_skip.png`](figures/group_scan_skip.png)
- 全コマンド、failure、再開点: [`RESEARCH_STATE.md`](../RESEARCH_STATE.md)

immutable snapshot は local path と集計時刻を除いた決定的 projection で、184 raw shard それぞれの
run-relative path、rows、bytes、SHA-256 を含む。`manifest.json` は 1,002,420 bytes と SHA-256
`ac6eb0cf49371588690aec15ea416dd7421e665525bb036f4949081e08d1e72c` を記録する。大容量 raw、download
corpus、index、`.venv` は研究ルールに従い Git へ commit しない。

検証環境は Ubuntu 22.04.5 LTS、CPython 3.10.12、NumPy 2.2.6、Faiss CPU 1.15.0、Xeon Gold
5416S、RAM 29 GiB。`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` thread はすべて 1。GPU は使っていない。

phase commit `7bfebe935257da8d6c427e52430f35c156e3c50b` を detached fresh worktree に checkout し、
新規 `.venv` で次を実行した。

| clean command | 結果 |
|---|---|
| `make setup` | sandbox 内の初回は DNS 制限で失敗し、その失敗を保持。network 許可で同じ command を再実行し成功。環境検査時の Git status は空、Faiss smoke passed、thread=1 |
| `make test` | 132 passed、1 deselected、8 warnings、8.49 s |
| `make smoke` | run `offline-smoke-20260905T122801.383599Z-43d3397eca`、8/8 blocks、384 raw rows、completed |
| `make report REPORT_INPUT=... REPORT_OUTPUT=.cache/final-report-analysis` | final 2 completed / 1 excluded、112,800 rows、violation 0、oracle 16/16。immutable summary SHA-256 は元と同じ `ac6eb0cf...d1e72c` |
| 追加 `make test-full` | 133 passed、8 warnings、54.55 s |

clean smoke の raw directory は主 worktree の `results/smoke/` に複製して保持した。初回 setup failure は
scientific test failure ではなく sandbox DNS の外部制約だが、成功へ書き換えず
[`RESEARCH_STATE.md`](../RESEARCH_STATE.md) に両試行を記録した。

## 17. 提出チェック

- [x] 実行していない工程を passed と呼んでいない。
- [x] final enum は H1/H2/H3、negative conditions、break-even と整合する。
- [x] ordinary L2 と Faiss squared-L2 を区別した。
- [x] strict `LB > tau-beta`、same frozen `C`、exact tie ordering を確認した。
- [x] 採用 run の completion/identity/raw hash/row count を再検証した。
- [x] incomplete/pre-audit/failure raw を削除せず、主値からの除外理由を残した。
- [x] single-query latency、capacity estimate、measured batch QPS を区別した。
- [x] paired key、query 再標本化、bootstrap seed、2,000 回を保存した。
- [x] build/assignment/memory/Receipt を無料扱いせず、ACID throughput と区別した。
- [x] test query を center、group 数、beta の tuning に使っていない。
- [x] 新規性、普遍的有効性、production safety を過大主張していない。
- [x] fresh worktree の setup -> test -> smoke -> report を固定 phase commit で再検証した。
