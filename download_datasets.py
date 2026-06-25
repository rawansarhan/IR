import ir_datasets

DATASETS = ["beir/quora/test", "beir/webis-touche2020"]

for ds_name in DATASETS:
    print(f"Downloading/loading: {ds_name}")
    ds = ir_datasets.load(ds_name)

    # أول مرة بتمرّ على docs/queries/qrels بتنزّل البيانات
    docs = list(ds.docs_iter())
    queries = list(ds.queries_iter())
    qrels = list(ds.qrels_iter())

    print(f"Done: docs={len(docs)}, queries={len(queries)}, qrels={len(qrels)}")
    print("-" * 40)
