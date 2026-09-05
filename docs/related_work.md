# 先行研究・隣接方式の一次資料調査

確認基準日: **2026-09-05 (Asia/Tokyo)**

## 調査範囲と読み方

本書は Issue #1 の工程9に対応する先行研究差分表である。Issue に指定された
[R1--R9](https://github.com/wasanemon/Searchability-Obligations/issues/1) をすべて確認し、
さらに Oracle 型 journal、delta ANN、metric/ball-based branch-and-bound、ANN の誤差保証、
query ごとの certificate、時間・version ベースの bounded staleness に近い一次資料を確認した。
一次資料とは、公式文書、公式ソースコード、原論文または著者配布の原論文を指す。

ここでいう「本案」の比較対象は、同じ query `q`、snapshot `s`、base generation `g` と、
visibility 処理後に一度だけ凍結した同一の base ANN 候補集合 `C` に対する次の契約である。

```text
R_ref  = TopK(C union Delta_s,g)
R_prop = TopK(C union S),  S subseteq Delta_s,g
0 <= tau_prop - tau_ref <= beta
```

`tau` と `beta` は通常の Euclidean 距離の単位である。group `G` の固定 center `c` と
全 member を覆う radius `r` から `LB(q,G) = max(0, d(q,c)-r)` を作り、
`LB > tau-beta` のときだけ group を省略する。省略 group の最小 safe lower bound を `L_min` とすると、
実数モデルでの事後上界は `max(0, tau_prop-L_min)`（省略なしは 0）である。実装の Receipt は
通常距離の外向き上界 `T_exact_hi` を用いた `certified_beta=max(0,T_exact_hi-L_min)` を権威値とする
（Issue の `beta_certified` に対応）。
したがって、この契約は DB 全体の exact top-k、
最新再構築 ANN との同一性、ID recall、全順位の誤差、更新後の経過時間を保証しない。

表の「確認箇所」は、2026-09-05 に実際に本文を読んだ節・関数である。「確認できず」は
その資料の当該確認範囲で見つからなかったという意味に限り、世界に存在しないという主張ではない。
Faiss Wiki、GitHub の `main`、製品文書は可変なので、再調査時には revision/commit と本文の再照合が必要である。

## Issue 指定資料 R1--R9

| ID | 一次資料・URL | 確認日・確認箇所 | 資料が与える保証・確認事実 | 検索・更新の仕組み | 本案との差 | 同等性の未確認点 |
|---|---|---|---|---|---|---|
| R1 | Faiss Wiki, [MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) | 2026-09-05; `METRIC_L2`, `How can I index vectors for cosine similarity?` | Faiss の L2 出力は **squared Euclidean distance** であり、通常 L2 が必要なら平方根を取る。正規化 vector では squared L2 が `2-2<x,y>`。 | 距離の表現・metric の変換規約。 | 本案の `radius/LB/tau/beta` は通常 L2。Faiss 出力をそのまま使うと単位が異なる。この資料自体は freshness や `beta` を保証しない。 | 使用中の `faiss-cpu==1.15.0` における各演算の丸め誤差、tie、overflow の厳密な上界は Wiki だけでは未確認。保証経路は別途 source と数値解析が必要。 |
| R2 | Faiss Wiki, [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) | 2026-09-05; `Summary of methods`, `IndexHNSW variants` の parameters / supported operations | 表では `IndexFlatL2` を exhaustive/brute-force、`IndexHNSWFlat` を non-exhaustive とする。HNSW は `M`, `efConstruction`, `efSearch` を持ち、vector removal は graph を壊すため非対応と明記。 | Flat は全 vector を比較。HNSW は多層近傍 graph を探索する。 | 本案では不変 HNSW generation を base、Flat を exact/full-scan baseline にする。Faiss HNSW の ANN miss と、Delta group 省略の追加誤差は別物。 | version 固定の API/tie 挙動、incremental `add` 後の品質、永続化・generation publish・snapshot visibility はこのページでは未確認。 |
| R3 | Faiss Wiki, [Faiss building blocks: clustering, PCA, quantization](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization) | 2026-09-05; `Clustering`, `Additional options`, `Assignment` | `Kmeans`/`Clustering` と seed 等の option、`kmeans.index.search(x,1)` による nearest-centroid assignment を示す。返る `D` は squared L2。 | 学習 data から centroid を作り、vector を最近 centroid へ割り当てる。 | 本案は base/学習用 data だけで center を作って固定し、Delta を割り当てた後に全 member を覆う conservative radius を別途保持する。k-means 自体は LB、snapshot、`beta` を保証しない。 | pinned version の再現性、複数 seed/初期化の差、center 学習と test/Delta の漏洩防止、radius の安全な丸めは別途検証が必要。 |
| R4 | Faiss Wiki, [Threads and asynchronous calls](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls) | 2026-09-05; `Thread safety`, `Internal threading` | CPU index は concurrent search など非変更操作について thread-safe。index を変更する操作の並行利用には利用側の mutual exclusion が必要。GPU index は read-only でも同じ扱いではない。 | CPU 内部では OpenMP/BLAS を利用し、呼出側が thread 数を制御する。 | 本案が search 中の generation を不変にし、別 generation を構築して publish/pin する理由を支持するが、Faiss は MVCC や obligation coverage を提供しない。 | `faiss-cpu==1.15.0` の個別 index 型・serialization 中の concurrency、Python wrapper の lifetime、publish/GC protocol は未確認。 |
| R5 | Faiss Wiki, [Brute force search without an index](https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index) | 2026-09-05; `Brute force search on CPU`, `Combining the results from several searches` | CPU の `knn_L2sqr` / `knn_inner_product`、Python の `contrib.exhaustive_search.knn`、slice ごとの結果を統合する `ResultHeap` を示す。 | raw matrix を直接全走査し、必要なら chunk ごとの top-k を heap で統合する。 | Delta full scan と全 visible data exact scan の最適化された基準実装に使えるが、省略、freshness、certificate は与えない。 | installed wheel での低水準関数の可用性、入力 dtype、tie、NaN、数値誤差、proposal と同程度に最適化された公平な計時は実測が必要。 |
| R6 | Oracle Database 26ai, [HNSW Index Architecture: Transaction Support and Persistence](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/hnsw-index-architecture-transaction-support-and-persistence.html) | 2026-09-05; `Transaction Support Structures`, `Graph Refresh and Persistence Structures`, `Explicitly Specifying a Graph Refresh` | memory-only HNSW には後続 DML が直接反映されない。private journal は transaction 中の insert/delete、table-backed shared journal は commit SCN と inserted/deleted VID の履歴を保持する。文書は shared-journal vector の検索を **exact**、graph 検索を approximate と区別し、transactionally consistent top-K を提供すると説明する。古すぎる snapshot は `ORA-51815`。 | query は graph と build 後の DML を併用。incremental snapshot は insert を graph に加え delete bitmap を追跡し、full repopulation 中も旧 graph を稼働させる。checkpoint は topology/metadata を保存し vector 自体は保存しない。 | 「immutable/stale graph + committed journal を query 時に統合」は既知であり、本案の新規性ではない。確認文書には group LB による journal 省略、query 指定 `beta`、同一 `C` に対する追加誤差、per-query certificate は記載されない。 | journal の exact scan/merge の具体的 algorithm、candidate 数と target accuracy の相互作用、境界/tie、内部 pruning、SCN ごとの GC/pinning、関連 patent・別節に同等 certificate があるかは未確認。 |
| R7 | TEXMEX/IRISA, [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)（Issue 記載の [HTTPS URL](https://corpus-texmex.irisa.fr/) も確認） | 2026-09-05; `Overview`, file formats, `Details and Download`, groundtruth/recall, license/footer、および公式 FTP `MD5SUM` | SIFT1M は 128 次元・base 1,000,000・query 10,000・learn 100,000、GIST1M は 960 次元・base 1,000,000・query 1,000・learn 500,000。`.fvecs/.ivecs` 等は little-endian。SIFT1M/GIST1M の ground truth は k=100、squared Euclidean 順。CC0 dedication を表示。これは dataset の事実であり algorithm 保証ではない。 | base/query/learning と precomputed exact-neighbor IDs を配布する。 | 評価 data source。base/Delta を独自に分割すれば、配布 ground truth の母集団と変わるため実際の全 visible 集合で再計算する必要がある。 | Issue の HTTPS endpoint は確認時に証明書名不一致、歴史的な HTTP archive 直下 URL は 404 だったため、ページが示す公式 FTP を使用した。SIFT/GIST は公式 MD5 と照合済みで、SHA-256 と展開各ファイルは `data/manifests/` に保存した。配布 ground truth は独自 split には流用しない。 |
| R8 | ANN-Benchmarks, [`ann_benchmarks/datasets.py` on main](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) | 2026-09-05; `get_dataset`, `write_output`, `_load_texmex_vectors`, `_get_irisa_matrix`, `sift`, `gist`, `DATASETS` | `sift`/`gist` は TEXMEX FTP archive から base/query `.fvecs` を読む。`write_output` は Euclidean 用 HDF5 を作り、brute-force で neighbor/distance を生成する。`DATASETS` に `sift-128-euclidean`, `gist-960-euclidean` を登録。 | 配布形式を parse し、benchmark 共通 HDF5 と exact baseline を構築する reference loader。 | 取得・format の参考であり、custom base/Delta split の正解や本案の保証ではない。公式配布 ground truth をそのまま流用する理由にもならない。 | `main` は可変で commit SHA を未固定。例外処理、各 row の dimension field を読み捨てる実装、checksum 検証、同じ preprocessing を本 artifact が採用するかは別途確認が必要。 |
| R9 | SQLite, [Atomic Commit In SQLite](https://www.sqlite.org/atomiccommit.html) | 2026-09-05; §§1--4、特に 2, 3.7, 3.10, 3.11, 4.2--4.6、および §§8--9 | rollback mode では、transaction 内の変更が全て見えるか全く見えないようにし、OS crash/power failure 時も atomic に見せる。journal を DB page より先に flush し、hot journal を回復時に replay する。文書は filesystem/VFS、`fsync`、atomic-looking file deletion 等の hardware assumptions と故障例も明示する。 | rollback journal に変更前 page を保存・flushし、DB を書いて flush、journal の削除/無効化を commit point とする。 | vector version と Obligation を同じ SQLite transaction に保存する基盤に使える。外部 Faiss index file、manifest rename、in-memory registration までを SQLite transaction が自動的に原子化するわけではない。 | artifact が使う journal mode/`PRAGMA synchronous`/VFS/filesystem、WAL mode の別機構、process-kill と実電源断の差、external file の fsync/rename/publish protocol は別途試験・文書化が必要。 |

## 最も近い研究・システムとの比較

| 分類・一次資料 | 確認日・確認箇所 | 保証 | 仕組み | 本案との差 | 同等性の未確認点 |
|---|---|---|---|---|---|
| **Oracle 型 journal** — R6 の [Oracle 26ai HNSW architecture](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/hnsw-index-architecture-transaction-support-and-persistence.html) | 2026-09-05; transaction support / graph refresh 全文 | transaction visibility と journal vector の exact search。base graph 自体は approximate。古い snapshot を無制限に受ける保証ではなく `ORA-51815` がある。 | private/shared journal、SCN、ROWID--VID map、graph + DML query、incremental snapshot/full repopulation/checkpoint。 | base+journal 併用と exact journal scan は直接の既知例。本案は journal 全走査を reference とし、その一部だけ省略した追加 kth-distance 影響を query ごとに証明しようとする。 | 前表 R6 の未確認点に同じ。Oracle の別文書・実装・patent に impact bound/certificate がないとは結論していない。 |
| **Delta ANN** — Singh et al., [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://suhasjs.github.io/files/freshdiskann-arxiv.pdf) (arXiv:2105.09613) | 2026-09-05; §1 fresh-ANNS 定義・Definition 1.1、§§5.1--5.3 components/API/StreamingMerge、§5.6 crash recovery、§6 evaluation | query result は時刻 `t` の active dataset と、完了済み insert/delete のある total order に整合する **quiescent consistency** を対象とする。品質は query 集合上の平均 `k-recall@k` で評価し、論文は `>95% 5-recall@5` 等を報告する。任意 query の加法/乗法距離上界ではない。 | SSD の Long-Term Index、recent update を持つ一つ以上の in-memory FreshVamana TempIndex、DeleteList を全て query して merge/filter。RW を RO snapshot にし、StreamingMerge で LTI へ反映。redo log と snapshot で crash recovery。 | 強い delta-ANN baseline。Delta を近似 graph で検索するため、exact Delta reference に対する見落としを本案の center-radius rule のように query ごとに certificate しない。一方、更新 throughput、delete、merge、recovery は本案より本格的。 | arbitrary historical snapshot/MVCC、DB transaction との commit atomicity、query ごとの距離保証、公開後の実装がこの版の論文と同じかは未確認。 |
| **Delta index と global rebuild の既知性／in-place 代替** — Xu et al., [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/pdf/2410.14452v1), [DOI](https://doi.org/10.1145/3600006.3613166) | 2026-09-05; abstract、§1、§§2.1--2.4、とくに `Out-of-place update` | 論文の品質尺度は `RecallK@K`。goal は new vector が高確率で recall されることで、任意 query の deterministic distance bound ではない。 | 既存の out-of-place 系は update を secondary in-memory index に蓄積し main と両方検索、定期 global rebuild。SPFresh は partition split/merge と境界 vector の局所 reassignment (LIRE) で in-place 更新する。 | 「delta index を別に持ち main と merge」は既知。本案は flat groups に全 Delta を保持し、query ごとに安全な group だけ省略する。SPFresh は steady-state に同じ obligation Delta を置く設計ではない。 | crash durability、MVCC、historical snapshot、LIRE の現在の製品実装、論文が言及する各製品の最新版挙動は本確認範囲外。 |
| **Metric-tree branch-and-bound** — Ciaccia, Patella, Zezula, [M-tree: An Efficient Access Method for Similarity Search in Metric Spaces](https://www.vldb.org/conf/1997/P426.PDF), VLDB 1997 | 2026-09-05; §2、§3.1、§3.2.2 とくに k-NNSearch/ChooseNode、§3.3 | metric と covering radius が正しければ exact k-NN。subtree の `d_min = max(d(q, center)-radius,0)` が current kth distance `d_k` を **strict に超える**とき安全に prune できる。 | routing object + covering radius の階層 tree。lower bound 最小の subtree を priority queue から選び、`d_k` を縮めながら branch-and-bound。dynamic insert/split も扱う。 | 本案の LB 式、LB 昇順、`beta=0` の strict prune は実質的に同じ既知原理。本案は full dataset の階層 tree ではなく、固定 base ANN 候補 `C` と flat Delta groups を合成し、threshold を `tau-beta` に緩和して obligation/snapshot/Receipt と結ぶ。 | relaxed additive rule と durable journal を一体化した既存実装、floating-point-safe bound、delete/GC（原論文は deletion 詳細を省略）は未確認。 |
| **Ball-tree branch-and-bound** — Stephen M. Omohundro, [Five Balltree Construction Algorithms](https://steveomohundro.com/wp-content/uploads/2009/03/omohundro89_five_balltree_construction_algorithms.pdf), ICSI TR-89-063 (1989) | 2026-09-05; `Balltrees`, `Queries which use Simple Pruning`, `Queries which use Branch and Bound` | node ball が子孫を包含するとき、query-centered current-best ball と交差しない node を exact NN/k-NN 探索から prune できる。近い child を先に検索する。 | center/radius の ball を binary tree にし、node ball までの minimum distance を子孫距離の lower bound として branch-and-bound。 | center と covering radius による安全な省略は古典的。本案の差は Delta に限定した flat grouping、加法 budget、同一 `C` comparator、snapshot/lifecycle contract。 | report の floating point/境界仕様、現代的高次元での性能、`beta>0` と per-query certificate の同等方式は未確認。 |
| **理論 ANN の乗法誤差** — Arya et al., [An Optimal Algorithm for Approximate Nearest Neighbor Searching in Fixed Dimensions](https://www.cse.ust.hk/faculty/arya/pub/JACM.pdf), JACM | 2026-09-05; abstract、Theorem 1、§1 algorithm overview、§§4--5 | 固定次元の Minkowski metric で `(1+epsilon)`-approximate NN と k 個の近似近傍を保証。`O(dn)` space、`O(dn log n)` build、dimension/epsilon 依存の logarithmic query bound を与える。 | BBD-tree の cell を query から近い順に priority searchし、次 cell の距離が current best / `(1+epsilon)` を超えれば停止。 | full static dataset の true NN に対する**乗法**保証。本案は高次元 HNSW の固定候補 `C` を受け入れ、Delta 省略だけの kth distance に**加法** `beta` を与え、worst-case sublinear query time を主張しない。 | 大規模高次元での実用性、finite precision、MVCC/update lifecycle は本案との同等性なし。論文は auxiliary structure で update 可能と述べるが、本確認では transactional semantics まで追っていない。 |
| **stream k-NN の加法誤差** — Nick Koudas, Beng Chin Ooi, Kian-Lee Tan, Rui Zhang, [Approximate NN Queries on Streams with Guaranteed Error/performance Bounds](https://www.vldb.org/conf/2004/RS21P3.PDF), VLDB 2004 | 2026-09-05; abstract、Definitions 1--2、§3.1 Theorem 1、§4 cell merging | `k <= K` で、`tau_true <= tau_footprint <= tau_true + d_M`。ここで `d_M` は cell 内の二点間最大距離（等分 Euclidean grid では `sqrt(d)/u`）であり、kth-distance の**加法上界**を明示する。 | unit hypercube を grid cell に分け、各 cell に最大 `K` footprint points を保持し、それらへ exact k-NN。DISC は B*-tree/Z-order と cell merge で memory/error を調整し、論文は sliding-window 版も提示する。 | 加法的 kth-distance 誤差という主張自体は既知。本案は stream を lossy synopsis にせず Delta 全件を耐久保持し、query と current `tau` に応じて group を省略、固定 `C + full Delta` を reference にして `certified_beta` を返す。 | 同論文の sliding-window 版と本案の version delete/MVCC の詳細対応、この系統の後続研究、per-query adaptive error、durable journal と組み合わせた方式、patent は未網羅。最優先の追加調査領域。 |
| **HNSW の実用品質保証の位置付け** — Malkov & Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320); Faiss, [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) | 2026-09-05; HNSW 原論文 §§3--4 の algorithms/recall experiments、Faiss `Do you need exact results?` と HNSW guidance | HNSW 原論文は recall--latency を実験し、`ef` で候補探索幅を調整する。確認範囲で任意 query の additive/multiplicative distance theorem は確認できない。Faiss 公式 guidance は exact を保証できる index は Flat とし、HNSW は `efSearch` による speed--accuracy trade-off とする。 | randomized multi-layer proximity graphを上層から greedy に辿り、base layer では bounded candidate list を探索。 | base HNSW 誤差は本案の契約外で別測定する。本案の `beta` を HNSW の recall/`efSearch` と合算したり、DB exact 誤差と呼んではならない。 | HNSW variant ごとの理論解析、特定 data 分布の確率保証、Faiss 実装版の細部は未網羅。「保証が存在しない」と一般化はしない。 |
| **query ごとの exact certificate** — Francis-Landau & Van Durme, [Exact and/or Fast Nearest Neighbors](https://matthewfl.com/papers/mfl.durme.nearest-neighbors-with-certificates.pdf), 2019 | 2026-09-05; §§1.1--1.2、§2、§3、§4 search integration、§5 evaluation | C2 は未確認領域に反例がないことを証明できた query では NN が exact という certificate を作る。budget 内に証明できなければ best guess（uncertified）または linear scan を選ぶ。top-k への一般化も説明する。 | unit vector/cosine、exact KNN graph、query-specific geometric constraints と feasibility solver を graph search に統合。 | per-query certificate そのものにも先行例がある。ただし C2 は静的 full dataset の exact-set certificate、本案は Delta groups の lower-bound frontierから fixed-reference に対する加法的 kth-distance upper bound を返す。 | update/durability/snapshot、Euclidean finite-precision certificate、C2 の top-k 実装範囲、本案の Receipt と同等の監査情報は未確認。 |
| **決定論的 bounded staleness の製品例** — Microsoft, [Azure Cosmos DB consistency levels](https://learn.microsoft.com/en-us/azure/cosmos-db/consistency-levels) | 2026-09-05; `Scope of read consistency`, `Guarantees associated with consistency levels`, `Bounded staleness consistency` | region 間 data lag を item の `K` versions または `T` time interval（先に達する方）未満に保つ。bound 超過時は partition の write を throttle。check は region 間で、region 内ではない。 | replication と replica read quorum により、各 region の最新 available version を configured lag 内で返す。 | `K/T` は「どれだけ古い version を読むか」、本案の `beta` は「未反映 vector の省略が回答距離へ与える影響」。age が大きくても検索結果影響 0、age が小さくても nearest update を落とせば影響大なので直交する契約。 | Cosmos DB の vector index/search がこの consistency level をどう継承するか、ANN index refresh lag まで同じ bound かは未確認。一般 item-read 文書から vector semantics を推測しない。 |
| **確率的 bounded staleness** — Bailis et al., [Probabilistically Bounded Staleness for Practical Partial Quorums](https://www.vldb.org/pvldb/vol5/p776_peterbailis_vldb2012.pdf), PVLDB 2012 | 2026-09-05; abstract、§§1.2, 3.1, 3.4、Definitions 1/3、§4 | PBS は partial quorum で last `k` versions 内を読む確率 (`k`-staleness) と、commit 後 `t` でその write 以上を読む確率 (`t`-visibility) を分析する。決定論的 bound を enforce する新機構ではないと明記。 | read/write quorum の交差確率と write propagation latency (WARS model) から staleness 分布を予測する。 | freshness/recency の確率であって、近傍結果の距離影響ではない。本案の deterministic per-query `beta` certificate と成功確率や平均 recall を混同しない。 | vector search への適用、ANN refresh、query-result impact との変換は扱わない。 |

## 同値・非同値の整理

### 既知と判断できる構成要素

1. **base ANN と未反映 journal/delta の併用は既知である。** Oracle は HNSW と committed DML journal を検索し、FreshDiskANN は long-term index と複数 TempIndex を検索して統合する。したがって「更新を別集合に置き、query 時に main と merge」だけを新規性にできない。
2. **center、covering radius、triangle inequality による lower bound は既知である。** M-tree の式は本案の `LB=max(0,d(q,c)-r)` と同じであり、current kth distance を使う best-first branch-and-bound と strict `>` pruning も既知である。本案の `beta=0` kernel は、その原理を flat Delta groups に適用したものと評価すべきである。
3. **kth nearest-neighbor distance の加法誤差保証も既知である。** Koudas--Ooi--Tan--Zhang (2004) は cell diameter による additive bound を定理として示す。誤差が additive であることだけ、本案が stream/update を扱うことだけでは差別化できない。
4. **近似探索の query-specific certificate も一般概念として既知である。** C2 は証明できた query だけ exact とする。certificate 失敗時に scan/fallback する考え方にも先例がある。
5. **ANN の一般保証と freshness は別軸である。** BBD-tree の `(1+epsilon)`、HNSW の empirical recall、Cosmos/PBS の `K/T` staleness はそれぞれ対象が異なり、本案の `beta` と置換できない。

### 現時点で残る可能性がある差分

確認した一次資料だけから残る差分候補は、個々の数式や journal ではなく、次の**限定された組合せと契約化**である。

- snapshot/generation ごとの durable Obligation を保持し、raw pending を含む visible Delta を定義すること。
- base ANN の品質を固定候補集合 `C` に封じ込め、Delta 省略だけによる追加 kth-distance 誤差を独立に比較できること。
- query 指定の absolute budget `beta` に対し、safe lower bound と strict boundary 判定を使い、実誤差ではなく安全上界 `certified_beta` と scope を Receipt に残すこと。
- update/delete、generation publish/pin、crash recovery、conservative GC とその certificate scope を一つの reference lifecycle に接続すること。

ただし、これは**新規性の立証ではない**。上記の組合せが論文・製品・特許に存在しないことは、本調査範囲からは確認できない。また、数学部分は M-tree の exact branch-and-bound を `tau-beta` に緩和した直接的な適用と評価される可能性が高い。

## 報告で混同してはならない量

| 量 | 比較先 | 典型資料 | 本案との関係 |
|---|---|---|---|
| `beta`: kth ordinary-L2 の additive impact | 同一 `C` + full visible Delta | 本案、Koudas et al. の additive distance bound（ただし母集団・機構は異なる） | 主契約。ID 一致や DB exact ではない。 |
| `(1+epsilon)`: multiplicative ANN error | full dataset の true NN/kth NN | Arya et al. BBD-tree | 別契約。距離 scale と comparator が異なる。 |
| recall@k / target accuracy | exact IDs に対する集合一致率 | HNSW、FreshDiskANN、SPFresh | 実測品質。query ごとの deterministic upper bound ではない。 |
| `K` versions / `T` time staleness | 最新 committed version | Cosmos bounded staleness、PBS | 更新の年齢・可視性。検索回答への幾何学的影響を直接は表さない。 |
| exact journal scan | journal 内の全 visible vector | Oracle | 本案の主 reference に近いが、base HNSW の ANN error は残る。 |
| certificate | 資料ごとに異なる命題 | C2、本案 | 「何を証明したか」を scope と comparator 付きで記載しない限り同義ではない。 |

## 2026-09-05 時点の研究判断

新規性の結論は **INCONCLUSIVE** である。確認済み事実からは、次の弱い表現までが妥当である。

> 既知の base+journal/delta 構成と、既知の metric/ball branch-and-bound を、固定 base ANN 候補に対する未反映 Delta 省略の影響契約として組み合わせ、数値的に安全な query-specific bound と durable lifecycle を一体で検証する研究プロトタイプ。

「初の journal-aware ANN」「初の additive-error stream k-NN」「中心半径枝刈りの新規発明」「初の per-query certificate」とは報告できない。性能上の有効性も本書では判断せず、保存された同一条件の実験結果だけから判断する。

## 未完了の新規性調査

本 Issue の prototype 完成を止めず、publication/patent claim の前に少なくとも次を追加確認する。

1. Koudas et al. (2004) の sliding-window 節と引用・後続研究を辿り、dynamic insert-delete stream k-NN、adaptive per-query additive bounds、synopsis を使わず検索を省略する方式を調べる。
2. M-tree、ball-tree、cover tree、VP-tree、metric index の approximate/range-relaxed variants と、frontier lower bound から返す a posteriori certificate を調べる。
3. Oracle AI Vector Search の関連 manual、公開 patent、SQL syntax/target accuracy、SCN consistency と journal merge の実装資料を調べる。
4. FreshDiskANN/DiskANN、SPFresh/SPANN、Milvus 等の最新版 source と publication を commit/version 固定で調べ、delta の exact/approximate search、visibility、crash recovery、historical snapshot を対応付ける。
5. floating-point filter、interval arithmetic、verified nearest-neighbor search の文献を調べ、本案の numerical certificate が既知手法の単純利用かを確認する。
6. 学術検索だけでなく patent database を、`vector index journal`, `delta index`, `bounded impact`, `certified nearest neighbor`, `covering radius`, `freshness` の組合せで調べる。これは法的な freedom-to-operate 判断とは別に専門家確認を要する。

## 取得上の注意

- R7 は 2026-09-05 の確認時、Issue にある HTTPS URL が certificate name mismatch で失敗したため、同じ host の HTTP ページを読み、そこに記載された公式 FTP から取得した。SIFT/GIST archive は配布元 MD5に一致し、artifact 側でSHA-256と展開各ファイルのSHA-256も `data/manifests/` に保存した。
- GitHub Wiki と `main` branch は後から変わる。再現実験で参照する source は commit/revision を manifest に固定する。
- 本書は source content の調査記録であり、ここに挙げた製品をローカルで実行したという記録ではない。実装・実験の実行済み範囲は `RESEARCH_STATE.md` と raw result manifest を正とする。
