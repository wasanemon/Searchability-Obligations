# Searchability Obligations / Impact-Bounded Freshness 検証報告

<!--
REPORT STATUS: SCAFFOLD
このファイルは人が研究判断を行う最終報告である。
reports/REPORT_ja.generated.md は保存済み raw data から再生成される集計素材であり、
このファイルを上書きしない。TBD を実測値で置換するときは run ID、config hash、
dataset hash / split ID と COMPLETED marker を必ず照合すること。
-->

- 報告基準日: 2026-09-05（Asia/Tokyo）
- 対象: [GitHub Issue #1](https://github.com/wasanemon/Searchability-Obligations/issues/1)
- 報告状態: **作業中の骨格。実データ最終評価前であり、性能値は未確定。**

## 0. 要旨と現在の判定

本研究は、immutable な Base HNSW に未反映の commit 済み更新を `Delta` として保持し、
固定 center と covering radius から得る下界によって Delta group を省略する方式を検証する。
保証対象は DB 全体の exact top-k ではない。同一 query、snapshot、base generation と、
一度だけ凍結した同一 Base 候補集合 `C` に対し、Delta の省略が k 番目の**通常 L2 距離**へ
追加する悪化だけを query 指定 `beta` 以下に抑える契約である。

現在の総合判定は **`INCONCLUSIVE`** とする。これは方式が失敗したという結論ではなく、
SIFT/GIST の完了済み最終 run、再監査後の性能比較、paired confidence interval、
break-even、clean-environment 再実行がまだ揃っていないための暫定判定である。
最終化時には、次の三つの enum のうち一つだけを残し、根拠となる run ID を併記する。

- `SUPPORTED_IN_TESTED_REGIME`
- `NOT_SUPPORTED_IN_TESTED_REGIME`
- `INCONCLUSIVE`

| 研究質問 | 現在の判定 | 最終判断に必要な証拠 |
|---|---|---|
| H1 / correctness | `INCONCLUSIVE`（最終一括再実行待ち） | 反例・境界・固定 seed 10,000 ケース、全 final raw row の契約検査 |
| H2 / usefulness | `INCONCLUSIVE` | SIFT/GIST での optimized Delta full scan との同一条件比較、paired 95% CI、負ける条件と break-even |
| H3 / lifecycle | `INCONCLUSIVE`（最終一括再実行待ち） | update/delete、snapshot、raw pending、process crash、generation publish/pin/GC の保存済みテスト結果 |

TBD（未実行）: 最終判定日、対象 commit、final run ID、結論の一文。

## 1. 仮説、方式、主張しないこと

### 1.1 研究仮説

- **H1 / correctness:** 同一の ANN 候補 `C` に visible Delta の全走査を加えた方式に対し、
  group 省略由来の k 番目距離の悪化を query 指定 `beta` 以内に保証できる。
- **H2 / usefulness:** 下界計算、group 管理、merge、Receipt 生成まで含めても、
  optimized Delta full scan より速くなる非自明な領域が存在する。
- **H3 / lifecycle:** update/delete、未 group 化データ、snapshot、クラッシュ、
  generation 切替があっても、visible version の取りこぼしや stale version の返却を防げる。

### 1.2 実装した範囲

- Python 3.10 / Faiss CPU の immutable `IndexHNSWFlat` Base と ordinal-to-version 対応。
- raw pending と immutable group からなる Delta、fixed center、conservative radius、連続 member 配列。
- Base 候補の visibility filtering・補充・凍結、Delta の full scan と certified group pruning、
  exact boundary ordering、構造化 Receipt。
- production 検索コードから独立した小規模 `Fraction` 全列挙 oracle。
- optimized Delta Flat、Delta HNSW、Base+Delta 全体 HNSW、全体 exact Flat、
  grouping/no-pruning ablation を含む比較 harness。
- SQLite の version/obligation 原子 commit、MVCC snapshot、raw 再構成、group catalog、
  immutable generation の publish/recover/pin と保守的 retention を扱う小規模参照 store。
- checkpoint 付き実験、query-method 単位の JSONL、manifest/hash、集計・描画。

上記は、最終 verification command の結果と整合することを提出前に再確認する。

### 1.3 実装・保証していない範囲

- 本番用 DBMS、SQL、分散 transaction、GPU、Faiss 内部改造、高並列 MVCC、独自 WAL。
- Base HNSW の miss を含む DB 全体 exact top-k 保証、最新再構築 HNSW との同一性。
- `beta > 0` での返却 ID、recall、全順位距離、経過時間 freshness の保証。
- Receipt の暗号学的自己検証性、任意の壊れた hardware/filesystem での耐久性。
- production 相当の write throughput、持続的 mixed workload、backlog 安定性。
- 積極的な version/obligation GC。参照実装は安全側に保持し、storage leak を許す。
- 画像 descriptor の結果を文章 embedding や RAG 全般へ一般化する主張。

## 2. 保証の契約

query `q`、MVCC snapshot `s`、pin した Base generation `g` を固定する。

```text
B_s,g       = g に入り、s で visible な version
Delta_s,g   = s で visible だが g に未反映の version（raw pending を含む）
C(q,s,g)    = Base ANN 候補を visibility 処理・重複排除・必要な補充後に凍結した集合
R_ref       = TopK(C union Delta_s,g)
R_prop      = TopK(C union S), S subseteq Delta_s,g
```

十分な候補があるとき、通常 L2 の k 番目距離について約束する量は次である。

```text
0 <= tau_prop - tau_ref <= beta
```

`C` は一度だけ生成し、その query/snapshot/generation、Base universe、version key と vector bits に
hash で binding した同じ object を reference と proposal に渡す。HNSW を二回検索して同じ `C` と
みなさない。Faiss の `METRIC_L2` が返す squared-L2 と、契約の ordinary-L2 を混同しない。

`beta = 0` では `(exact_squared_distance, logical_id, version_id)` の順で ordered top-k まで
reference と一致させる。`beta > 0` で保証するのは k 番目距離だけである。

## 3. Group rule、証明、数値契約

group `G` の固定 center `c` と、全 member を覆う上向き radius `r_hi` に対して、

```text
LB(q,G) = max(0, lower(d(q,c)) - r_hi)
skip G iff LB(q,G) > upper(tau) - beta
```

を使う。不等号は必ず strict `>` であり、`LB = tau - beta` は scan する。raw pending は
先に全走査し、group は `(LB, group_id)` 昇順で処理する。凍結後は候補を追加するだけなので
k 番目の上向き threshold は非増加であり、以前の skip 判定を壊さない。delete/visibility の
後処理で候補を除く方式はこの単調性を失うため禁止する。

skip した group の最小下界を `L_min` とすると、実誤差ではなく安全上界として

```text
certified_beta = max(0, tau_exact_hi - L_min)
```

を Receipt に返す。skip がなければ 0 とする。完全な証明、前提、fallback は
[`docs/correctness.md`](../docs/correctness.md)、浮動小数点の区間構成は
[`docs/numerics.md`](../docs/numerics.md) を正とする。

保証付き経路は canonical binary32 入力、次元 `1..4096`、各成分の絶対値 `<= 1e15`、
有限かつ非負の binary64 `beta` を受理する。距離計算では binary64 の演算誤差を
`gamma_(d+2)` で囲み、radius は上側、LB は下側、tau は上側に評価する。曖昧な最終境界は
binary32 成分を exact rational に変換して再順位付けする。NaN/Inf、complex、次元不一致、
範囲外、負の beta、不正 k は certified result を返す前に拒否する。

候補不足、保持 snapshot に対応する generation 不在、壊れた group bound、HNSW underfill は、
適用不能表示、全 visible exact、該当 group scan、Base 全列挙補充のいずれかへ安全側に fallback
する。epsilon-only または plain-float path を certified と呼ばない。

## 4. 正当性・反例・復旧テスト

最終報告では、次を「実装済み」ではなく**実行済み結果**として、コマンド、件数、経過時間、
JUnit/hash、失敗 case の保存先とともに記載する。途中の実行履歴は
[`RESEARCH_STATE.md`](../RESEARCH_STATE.md) に残す。

| 検査 | 最終結果 | 証拠 |
|---|---|---|
| `beta=0` の同一 `C` ordered top-k 一致 | TBD（最終再実行待ち） | TBD |
| `0 <= observed <= certified <= requested` | TBD（最終再実行待ち） | TBD |
| radius coverage と member ごとの LB | TBD（最終再実行待ち） | TBD |
| empty/singleton/tie/radius=0/LB 境界/beta>tau | TBD（最終再実行待ち） | TBD |
| underfill、k 超過、duplicate/multiversion/visibility/delete | TBD（最終再実行待ち） | TBD |
| 異なる scale、outlier、丸め境界、invalid input | TBD（最終再実行待ち） | TBD |
| 固定 seed 10,000 randomized cases | TBD（最終再実行待ち） | `results/test_evidence/` の final artifact を記載 |
| delete 後処理、latest-only snapshot、center 移動、早すぎる GC の反例 | TBD（最終再実行待ち） | TBD |
| 小 beta でも ID が多数入れ替わる反例 | TBD（最終再実行待ち） | TBD |
| insert/update/delete と obligation の process-crash atomicity | TBD（最終再実行待ち） | TBD |
| group/generation publish、pin、recover、conservative GC | TBD（最終再実行待ち） | TBD |

契約違反が 1 件でもあれば、その raw row と最小化した反例を保持し、性能値より先に原因を
修正する。修正不能なら H1 は支持しない。`beta` を事後的に広げて失敗を隠さない。

## 5. データ、分割、設定

### 5.1 データ

| データ | 用途 | 次元 | final 状態 | provenance |
|---|---|---:|---|---|
| synthetic clustered | 分離 cluster | config に保存 | final smoke TBD（未実行） | `configs/smoke.json` |
| synthetic isotropic | 高次元・弱い bound | config に保存 | final smoke TBD（未実行） | `configs/smoke.json` |
| synthetic delta-near-queries | query に関係する更新の stress | config に保存 | final smoke TBD（未実行） | `configs/smoke.json` |
| synthetic outlier-radius | radius 膨張の stress | config に保存 | final smoke TBD（未実行） | `configs/smoke.json` |
| TEXMEX SIFT | 実データ主比較 | 128 | 取得 manifest あり、final 評価 TBD（未実行） | `data/manifests/sift.json` |
| TEXMEX GIST | 高次元実データ主比較 | 960 | 取得 manifest あり、final 評価 TBD（未実行） | `data/manifests/gist.json` |

各 run は Base、Delta、validation query、test query の ID、dataset hash、split ID を保存する。
center は Base/指定 training split だけから作り、`beta`、group 数、改善方式は validation で決め、
final test query では調整しない。再分割した visible 集合の exact truth は再計算し、配布済み
ground truth を流用しない。`delta_near_queries` の意図的相関は通常分割と分離して表示する。

### 5.2 初期条件と一軸 sweep

初期条件は SIFT で Base 100,000、Delta 10,000、test 1,000、`k=10`、groups 128、
HNSW `M=32`、`efConstruction=200`、Base `efSearch=128`、candidate count 64、1 CPU thread とする。
GIST は memory 上限により Base 50,000、Delta 5,000、test 800 の縮小条件であり、SIFT と同規模を
実行したようには扱わない。

保存済み config は Delta size `0/1,000/10,000/100,000`、group count `16/64/128/512`、
`k=1/10/100`、validation 基準距離 scale に対する正の beta factor `0.01/0.05/0.10`、
複数 `efSearch`、3 build/split seeds、主要条件 3 repetitions を対象とする。実際に完了した条件だけを
最終表に含め、resource 制限で縮小した run は effective config hash とともに別扱いにする。

TBD（未実行）: final run ごとの絶対 beta、完了条件一覧、除外条件と理由。

## 6. 比較方式と公平性

| 方式 | 役割 | 主比較か |
|---|---|---:|
| Base HNSW + optimized Delta full scan | 同一 `C` を使う速度・正しさ reference | 主比較 |
| Base HNSW + group pruning, `beta=0` | 省略追加誤差 0 の提案方式 | 主比較 |
| Base HNSW + group pruning, `beta>0` | 速度と certified impact の trade-off | 主比較 |
| Base HNSW + Delta HNSW | Delta 側も ANN にする実用比較 | 補助比較 |
| Base+Delta 全体 HNSW | 反映済み方式の品質・検索・build/add cost | 補助比較 |
| Base+Delta exact Flat | 全 visible recall/distance truth | 正解計算、主速度勝負ではない |
| grouping, no pruning | group 管理・分割 scan overhead | 必須 ablation |

final run の公平性 checklist:

- [ ] 正しさ比較の全 coupled 方式が同一の frozen `C` object を使う。
- [ ] optimized Delta Flat は一回の full scan であり、検証用二重 scan を計時へ混入させない。
- [ ] 同じ ordinary-L2、canonical dtype、データ、`k`、Base index、candidate 数、thread 数を使う。
- [ ] 初期 Base `efSearch=128` を明示し、efSearch sweep では条件ごとに `C` を凍結し直す。
- [ ] Base 検索を除く coupled micro latency と、Base 検索込み E2E latency を分離する。
- [ ] oracle、ground-truth、download、validation 選択は query latency から除外する。
- [ ] visibility、raw scan、LB、sort、group scan、merge、Receipt の費用は除外しない。
- [ ] warmup 後に測定し、method 実行順を query/repetition ごとに決定的に入れ替える。
- [ ] Delta/full HNSW は recall-latency 曲線で比較し、低品質設定だけで勝敗を作らない。
- [ ] build/add/group/assignment/memory は別途計測し、ACID commit throughput と呼ばない。
- [ ] `COMPLETED.json`、raw shard 行数・SHA-256、run/config/dataset/split identity を再検証する。
- [ ] audit 前、incomplete、identity 不整合、契約違反 run を主結果から除外し、除外理由を残す。

## 7. 測定定義と統計手順

### 7.1 Latency と batch throughput

single-query E2E latency は各 query-method-repetition の wall time から p50/p95/p99 を報告する。
coupled micro は frozen `C` 後の Delta 部分だけであり、E2E には Base preparation を加える。
batch throughput は、同じ query batch を実際に処理した総 query 数 / batch wall time で測る。
`1 / mean(single-query latency)` は throughput と呼ばず、必要なら単なる reciprocal-latency estimate
として別名で表示する。

TBD（未実行）: final run の batch size、trial 数、warmup、measured QPS と CI。

### 7.2 Paired comparison と不確実性

主 speedup は、同じ `(dataset hash, split ID, experiment, query_id, repetition)` の
optimized Delta full scan と提案方式を pair にし、`t_full / t_prop`（1 より大きいほど提案が速い）
で定義する。method ごとに独立に集めた非対応平均の比は主根拠にしない。

最終集計では repetitions を query 内でまとめた後、query ID を再標本化単位とする paired
bootstrap を使う。全 method の対応を保ったまま 10,000 回再標本化し、bootstrap seed を
summary に保存して、median speedup と percentile 95% CI を示す。複数 seed をまとめる場合は
seed と query の階層を保持し、method row を独立標本として水増ししない。p99 は有限標本の
empirical tail であり、とくに GIST 800 query と smoke の p99 には強い不確実性がある。

TBD（未実行）: bootstrap 実装/seed、条件別 median speedup `[95% CI]`、noise sensitivity。

### 7.3 品質、certificate、work reduction

条件ごとに次を保存・報告する。

- `tau_prop - tau_ref` の安全な observed interval、`certified_beta`、requested `beta`、違反件数。
- full visible exact top-k に対する recall@k、ID symmetric difference、rank difference。
- same-`C` reference top-k に Delta が入る query の割合と、その部分集合だけの latency/quality。
- Delta vector read/scan 数と割合、group skip 率、fallback 率。
- LB、tau、radius の分布と、scan になった理由。
- HNSW の visited count は使用 API で信頼して取得できない場合 `N/A` とし、推測しない。

## 8. 自動集計を取り込む領域

`make report` は保存済み raw shard と checkpoint の hash/row count を再検証し、
[`reports/REPORT_ja.generated.md`](REPORT_ja.generated.md)、summary JSON/CSV、figure を生成する。
自動素材は研究判断を行わず、この手動報告を上書きしない。

<!-- BEGIN AUTO-GENERATED-DATA-SYNTHESIS -->

TBD（未実行）: final run だけを入力に再生成した `REPORT_ja.generated.md` から、以下を転記・要約する。
転記した各数値には run ID、experiment ID、method、config hash を対応付ける。

- 完了 run と raw row の完全性。
- 方式別 latency、measured batch throughput、quality、skip/fallback、certificate violation。
- LB/radius/tau 分布と Delta-influence subset。
- 使用した figure と生成元 summary hash。

<!-- END AUTO-GENERATED-DATA-SYNTHESIS -->

audit 前の smoke artifact は失敗ではなく開発履歴として保持するが、既知の計時・identity 監査が
終わる前の値を final comparison に混ぜない。最終化時に、採用 run と除外 run を明示する。

## 9. 性能・品質の最終結果テンプレート

### 9.1 主比較

| dataset / condition | method / beta | E2E p50/p95/p99 ms | measured batch QPS | paired speedup [95% CI] | exact recall@k | Delta-influence recall | skip / fallback | certificate violations |
|---|---|---|---|---|---|---|---|---:|
| SIFT initial | optimized Delta full scan | TBD（未実行） | TBD | reference | TBD | TBD | N/A | TBD |
| SIFT initial | pruning beta=0 | TBD（未実行） | TBD | TBD | TBD | TBD | TBD | TBD |
| SIFT initial | pruning beta>0 | TBD（未実行） | TBD | TBD | TBD | TBD | TBD | TBD |
| GIST reduced initial | optimized Delta full scan | TBD（未実行） | TBD | reference | TBD | TBD | N/A | TBD |
| GIST reduced initial | pruning beta=0 | TBD（未実行） | TBD | TBD | TBD | TBD | TBD | TBD |
| GIST reduced initial | pruning beta>0 | TBD（未実行） | TBD | TBD | TBD | TBD | TBD | TBD |

TBD（未実行）: Delta HNSW / full HNSW の recall-latency curve、full exact 参考 latency。

### 9.2 費用内訳と memory

| dataset / condition | Base search | visibility | raw scan | LB | ordering | group scan | merge | Receipt | vectors read | peak RSS / incremental bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIFT initial | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GIST reduced initial | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

peak RSS は process high-water mark で、同時 resident な比較 index や allocator の保持を含みうる。
可能な範囲で Base index、Delta Flat/HNSW、group center/radius/member、raw、build 一時領域の
bytes を分離する。Python object size を vector payload 全体の memory と誤認しない。

### 9.3 構築・維持費

| 項目 | SIFT | GIST | 測定上の意味 |
|---|---:|---:|---|
| Base HNSW build | TBD（未実行） | TBD | component build cost |
| Full HNSW build/add | TBD（未実行） | TBD | ACID commit latency ではない |
| Delta HNSW build/add | TBD（未実行） | TBD | 同上 |
| center training | TBD（未実行） | TBD | test query を使わない |
| Delta assignment | TBD（未実行） | TBD | 1 vector / batch の条件を明記 |
| group materialization/radius | TBD（未実行） | TBD | copy と validation を含む |
| extra resident / peak memory | TBD（未実行） | TBD | 計測定義を併記 |
| SQLite commit/group/publish | TBD（未実行または非対象） | TBD | 小規模 lifecycle reference のみ |

持続的 mixed workload を実行していない限り、write throughput や backlog の定常安定性は
`未測定` とする。background work を query latency から隠して throughput 改善と呼ばない。

## 10. Break-even と負ける条件

### 10.1 Query-side break-even

同一品質・同一 `C` で correctness 条件を満たし、paired speedup の 95% CI 下端が 1 を超える
最小 Delta size / group condition を query-side break-even と定義する。測定 grid の間を補間して
確定値のように扱わない。crossing がなければ `tested grid 内になし` と報告する。

TBD（未実行）: SIFT/GIST、`beta=0`/positive beta、group count ごとの crossing または不存在。

### 10.2 Maintenance amortization

提案方式だけに必要な追加構築費を `B_extra`、一 query 当たりの full-scan との差を
`Delta_t = t_full - t_prop` とすると、`Delta_t > 0` の条件だけ

```text
Q_break_even = ceil(B_extra / Delta_t)
```

を参考値として示す。`Delta_t <= 0` なら有限の amortization break-even はない。更新により
group rebuild が必要になる頻度を明記し、異なる durability/ack 条件を同じ費用として比較しない。

TBD（未実行）: 条件別 `B_extra`、`Delta_t`、`Q_break_even` と感度分析。

### 10.3 Negative / failure analysis

最終報告では、勝った条件だけでなく少なくとも次を分けて説明する。

| 想定原因 | 観察する証拠 | 最終所見 |
|---|---|---|
| sphere LB が弱い | LB/tau/radius、skip 率、isotropic/GIST | TBD（未実行） |
| outlier で radius が膨張 | radius tail、outlier stress | TBD（未実行） |
| Python/metadata/order overhead | component timer、profile | TBD（未実行） |
| 小 group の分割 scan が非効率 | no-pruning vs contiguous full scan | TBD（未実行） |
| 小 Delta では固定費が支配 | Delta-size sweep | TBD（未実行） |
| query に Delta が関係しないだけ | Delta-influence subset | TBD（未実行） |
| fallback が支配 | reason 別 fallback 率 | TBD（未実行） |
| 高次元で計算/memory が支配 | SIFT vs reduced GIST | TBD（未実行） |
| 共有 host noise / tail 標本不足 | repetitions、paired CI、p99 sample warning | TBD（未実行） |
| 契約・data・resource failure | failure manifest、raw counterexample | TBD（未実行） |

## 11. 改善と ablation

最初の fixed-center/sphere-bound 方式を削除・上書きせず、改善前の raw run を保持する。
改善は validation 側の profile と failure category から選び、test 結果を見て選ばない。

| variant | 分離する効果 | 選定根拠 | final 結果 |
|---|---|---|---|
| optimized Delta full scan | 主 reference | predetermined | TBD（未実行） |
| grouping, no pruning | group 管理・分割 scan overhead | required ablation | TBD（未実行） |
| initial sphere pruning | 単純 LB の純効果 | Issue #1 初期案 | TBD（未実行） |
| small-Delta full-scan fallback | 小 Delta の固定費回避 | validation/profile のみ | TBD（未選定/未実行） |
| packed/multi-group scan | 小 scan の実装費用 | validation/profile のみ | TBD（未選定/未実行） |
| outlier separation | radius tail 改善 | validation/profile のみ | TBD（未選定/未実行） |

`beta` を大きくして全 group を skip しただけの条件を主要成功としない。改善を実装しなかった
場合も、`未実装` と理由を残す。

## 12. Lifecycle の安全性

参照 store は logical ID と version ID を分離し、`begin_seq <= s < end_seq` の interval で
snapshot visibility を決める。vector version の insert/終了と obligation を同じ SQLite transaction
へ置き、commit 後・in-memory 登録前・再起動後も durable obligation から raw pending を再構成する。

group は copy-then-SQLite-catalog-publish、generation は一時 directory への file/fsync、rename、
self-validating manifest、SQLite publication の順で公開する。query は snapshot、generation、catalog
revision、Delta view を pin する。古い generation を必要とする query がある間は、最新 generation の
完成だけを理由に obligation/version を回収しない。

actual-process fault injection は単なる正常 restart ではなく、insert/update/delete transaction、
commit 後 grouping 前、group catalog switch、および generation の
index/metadata 作成、file/directory fsync、rename、manifest install、SQLite catalog publish の境界で、
子 subprocess を `os._exit(86)` させて回復結果を確認する。一方、旧 generation を pin した query view と
新 generation publish の交錯、および active pin 中の GC 拒否・pin 解放後の保守的 retention は、同一
pytest process 内で順序を制御した interleaving test であり、process-crash test ではない。前者は
process crash に対する証拠、後者は実装上の pin/GC protocol に対する証拠である。いずれも電源断、
controller cache、filesystem corruption 全般を実証するものではない。

TBD（最終再実行待ち）: lifecycle test count、実行時間、failure 数、artifact/command。

## 13. 先行研究、新規性、非同値性

一次資料と確認箇所の詳細は [`docs/related_work.md`](../docs/related_work.md) を正とする。
確認済みの重要点は次である。

- Base ANN と journal/delta の併用は Oracle や FreshDiskANN 等に先行例があり、新規性ではない。
- center/covering radius/triangle inequality による strict branch-and-bound は M-tree/ball-tree の
  既知原理である。
- k 番目距離の additive error、query-specific certificate にも先行研究がある。
- HNSW の empirical recall、BBD-tree の multiplicative error、時間/version bounded staleness は、
  本案の same-`C` Delta omission impact と異なる契約である。

現時点の新規性判断は **`INCONCLUSIVE`** である。限定的な差分候補は、durable obligation と
snapshot/generation を、same-`C` に対する query-specific additive impact certificate へ結び付けた
組合せと契約化にある。ただし先行論文・製品・特許に同じ組合せがないことは確認できておらず、
「初の journal-aware ANN」「初の additive stream k-NN」「初の per-query certificate」などとは
主張しない。性能上の支持と学術的新規性の確認は別の判断である。

## 14. 制約と妥当性への脅威

- SIFT/GIST は画像 descriptor であり、workload・更新時系列・embedding 分布を代表しない。
- 静的 corpus の Base/Delta 分割は、現実の時間順更新を再現しない。
- GIST は縮小条件、test 800 件であり、SIFT と同規模の比較や精密な p99 推定ではない。
- Base HNSW の誤差は主契約外。exact recall は測るが `beta` へ合算しない。
- 1 CPU thread、共有 host、Python/Faiss の一 version に限定される。
- build と query の component benchmark は production の concurrent mixed workload ではない。
- process-kill test は power loss や任意 filesystem/hardware fault の代用ではない。
- conservative retention により、長期運用時の storage/GC 費用は未解決である。
- paired CI は測定 noise と有限 query 集合の不確実性を示すが、未知 workload への外的妥当性を
  与えない。
- 先行研究調査は網羅的 patent/FTO 調査ではない。

TBD（未実行）: 実測で判明した追加 limitation、失敗・除外・欠測の一覧。

## 15. H1/H2/H3 の最終評価と次段階

| 仮説 | 支持/不支持/不明 | 根拠 run/test | 反証・例外 |
|---|---|---|---|
| H1 | TBD（未実行） | TBD | TBD |
| H2 | TBD（未実行） | TBD | TBD |
| H3 | TBD（未実行） | TBD | TBD |

総合判定: **`INCONCLUSIVE`（暫定）**。

`SUPPORTED_IN_TESTED_REGIME` とするには、H1/H3 の final suite と全 raw row に契約違反がなく、
少なくとも一つの非自明な実データ条件で optimized full scan に対する paired CI を伴う利得があり、
Delta-influence subset、品質、fallback、memory、maintenance cost を含めても説明可能であることを
根拠にする。測定範囲全般で公平に遅い場合は negative result として
`NOT_SUPPORTED_IN_TESTED_REGIME` を選べる。必須 run 不足、noise、相反する dataset 結果により
結論できない場合は `INCONCLUSIVE` を維持する。

本格 DBMS 実装へ進む合理性: TBD（未実行）。進む場合は対象 regime、必要な native kernel、
mixed workload、retention/GC、production durability の次実験を限定する。進まない場合も、
correctness artifact、negative performance result、break-even 不在を成果として明記する。

## 16. 再現手順と証拠の所在

標準手順は次である。各コマンドの実行済み状態、exact output、failure、次の resume command は
[`RESEARCH_STATE.md`](../RESEARCH_STATE.md) を正とする。

```bash
make setup
make test
make test-full
make smoke
make data
make evaluate
make report
```

- dependency pin: `requirements-lock.txt`, `pyproject.toml`
- environment/config: run ごとの `run_manifest.json`, `effective_config.json`
- data provenance: `data/manifests/*.json`
- split/validation selection: run ごとの `splits/*.json`, `validation/*.json`
- raw rows: run ごとの `raw/*.jsonl`
- checkpoint/completion: `checkpoint.json`, `COMPLETED.json`
- build/memory: `build_manifest.json`, `run_manifest.json`
- machine aggregate: `results/**/analysis/summary.json`, `summary.csv`
- generated Japanese material: `reports/REPORT_ja.generated.md`
- figures: `reports/figures/`
- final human judgment: 本ファイル

TBD（未実行）: final commit/dirty state、CPU/RAM/OS、Python/Faiss/BLAS、thread env、全 final run ID、
config/dataset/split/checkpoint/raw/report の SHA-256、clean-environment command results。

## 17. 提出前チェック

- [ ] 本ファイルの `TBD（未実行）` を、証拠がある項目だけ実値または明示的な未完了理由へ置換した。
- [ ] final enum は一つで、H1/H2/H3、負ける条件、break-even と矛盾しない。
- [ ] 数学記号は普通 L2、Faiss 値は squared-L2 と区別した。
- [ ] strict `LB > tau-beta`、same frozen `C`、exact tie ordering を確認した。
- [ ] 全採用 run に `COMPLETED.json` があり、raw hash/row count と identity を検証した。
- [ ] incomplete/pre-audit/failure run を削除せず、最終値からの除外理由を記載した。
- [ ] single-query latency と measured batch throughput を混同していない。
- [ ] paired CI の pair key、再標本化単位、seed、回数を保存した。
- [ ] build/assignment/memory/Receipt を無料扱いせず、ACID throughput と component cost を区別した。
- [ ] test query を tuning に使っていない。
- [ ] 新規性、普遍的有効性、production safety を過大主張していない。
- [ ] clean environment で setup -> test -> smoke -> report を実行し、結果を保存した。
