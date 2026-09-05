# Preserved pre-audit smoke evidence

`offline-smoke-20260905T050837.464036Z-c36b660d90` completed before the
independent measurement and immutable-payload audits. Its raw rows are retained
to preserve the first center/radius implementation result, but it is not final
scientific evidence because:

- the optimized Delta Flat path performed a duplicate certified distance pass;
- throughput was not yet measured as an actual sequential batch;
- synthetic split IDs did not yet bind vector content;
- source-tree identity and the later immutable-array regression were absent.

The run manifest predates `evidence_role`; the analyzer therefore labels it
`legacy_unspecified`. `make smoke` selects only `calibration`, and `make report`
selects only completed `final` real-data runs, so this run cannot enter either
current aggregate accidentally.
