"""Streamlit UI — Information Retrieval System."""



from __future__ import annotations



import pandas as pd

import requests

import streamlit as st



API = "http://localhost:8000"



MODEL_INFO = {

    "tfidf": {

        "name": "TF-IDF (VSM)",

        "matching": "Cosine Similarity",

        "ranking": "تنازلي حسب درجة التشابه (0 → 1)",

        "detail": "مطابقة vector الاستعلام مع vectors الوثائق باستخدام Cosine Similarity",

    },

    "bm25": {

        "name": "BM25",

        "matching": "BM25 Okapi Scoring",

        "ranking": "تنازلي حسب BM25 score",

        "detail": "TF + IDF + تطبيع طول الوثيقة (معاملات k1 و b)",

    },

    "embedding": {

        "name": "Embedding (BERT)",

        "matching": "Cosine Similarity على Dense Embeddings",

        "ranking": "تنازلي حسب التشابه الدلالي (0 → 1)",

        "detail": "all-MiniLM-L6-v2 — يفهم المعنى مش بس الكلمات الحرفية",

    },

    "hybrid_serial": {

        "name": "Hybrid Serial (تسلسلي)",

        "matching": "BM25 → ثم Embedding rerank",

        "ranking": "مرحلة 1: BM25 candidates → مرحلة 2: Cosine Similarity rerank",

        "detail": "BM25 يجيب أفضل 100 وثيقة، Embedding يعيد ترتيبها",

    },

    "hybrid_parallel": {

        "name": "Hybrid Parallel (تفرعي)",

        "matching": "BM25 + Embedding بالتوازي → Weighted Fusion",

        "ranking": "دمج الدرجات المُطبّعة بأوزان قابلة للتعديل",

        "detail": "كل نموذج يبحث مستقلًا ثم تُدمج النتائج",

    },

    "vector_store": {

        "name": "Vector Store (FAISS)",

        "matching": "Cosine Similarity عبر فهرس FAISS",

        "ranking": "تنازلي حسب التشابه الدلالي من FAISS",

        "detail": "بحث دلالي سريع باستخدام فهرس FAISS IndexFlatIP بدل ضرب المصفوفة الكامل",

    },

}



ALL_MODELS = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel", "vector_store"]





def fetch_full_doc(dataset: str, doc_id: str) -> str | None:

    r = requests.get(f"{API}/docs/{dataset}/{doc_id}")

    if r.ok:

        return r.json().get("text", "")

    return None





st.set_page_config(page_title="IR System", page_icon="🔍", layout="wide")



st.title("🔍 Information Retrieval System")



# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:

    st.header("⚙️ Settings")



    dataset = st.selectbox("📂 Dataset", ["touche", "quora"])

    limit = st.number_input(

        "Docs limit (0 = full dataset)",

        min_value=0,

        value=0,

        step=1000,

        help="للعرض التقديمي: استخدم 0 = كل البيانات",

    )

    limit_val = int(limit) if limit > 0 else None

    rebuild = st.checkbox("Force rebuild models", value=False)



    if st.button("🚀 Load Dataset", use_container_width=True):

        with st.spinner("Loading corpus → SQLite → Index → Pre-building all models..."):

            r = requests.post(

                f"{API}/load",

                json={"dataset": dataset, "limit": limit_val, "rebuild": rebuild},

                timeout=36000,

            )

        if r.ok:

            d = r.json()

            st.session_state["load_info"] = d

            st.success(

                f"✅ {d['docs_loaded']:,} docs | SQLite ✓ | Models pre-built ✓ | {d['elapsed_sec']}s"

            )

            steps = d.get("steps", {})

            if steps.get("model_timings"):

                st.caption("Model build times (sec):")

                for m, t in steps["model_timings"].items():

                    st.caption(f"  • {m}: {t}s")

            elif steps.get("models") == "loaded_from_cache":

                st.caption("Models loaded from disk cache.")

        else:

            st.error(r.text)



    load_info = st.session_state.get("load_info")

    if load_info and load_info.get("dataset") == dataset:

        st.info(f"📦 Loaded: {load_info['docs_loaded']:,} docs · {load_info['queries']} queries")



    st.divider()

    st.subheader("🔀 Search Mode")

    search_mode = st.radio(

        "Pipeline",

        ["Basic", "Enhanced"],

        help="Basic: استعلام مباشر | Enhanced: Query Refinement قبل البحث",

    )



    st.divider()

    st.subheader("🤖 Model")



    model = st.selectbox("Retrieval Model", ALL_MODELS, index=1)



    if model == "bm25":

        st.markdown("**BM25 Parameters**")

        bm25_k1 = st.slider("k1 (term saturation)", 0.5, 3.0, 1.5, 0.1)

        bm25_b = st.slider("b  (length normalization)", 0.0, 1.0, 0.75, 0.05)

    else:

        bm25_k1, bm25_b = 1.5, 0.75



    if model == "hybrid_parallel":

        st.markdown("**Fusion Weights**")

        bm25_w = st.slider("BM25 weight", 0.0, 1.0, 0.4, 0.05)

        emb_w = st.slider("Embedding weight", 0.0, 1.0, 0.6, 0.05)

    else:

        bm25_w, emb_w = 0.4, 0.6



    top_k = st.slider("Top-K results", 5, 50, 10, 5)



    st.divider()

    st.subheader("📐 Matching & Ranking")

    info = MODEL_INFO.get(model, {})

    st.markdown(f"**النموذج:** {info.get('name', model)}")

    st.markdown(f"**Matching:** `{info.get('matching', '-')}`")

    st.markdown(f"**Ranking:** {info.get('ranking', '-')}")

    st.caption(info.get("detail", ""))



# ── Main — Search ────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:

    query = st.text_input("🔎 Enter your query", placeholder="e.g. Should bottled water be banned?")

with col2:

    st.markdown("<br>", unsafe_allow_html=True)

    search_btn = st.button("Search", use_container_width=True, type="primary")



if query:

    with st.expander("🔬 Query Processing Steps"):

        r = requests.post(f"{API}/query-steps", json={"query": query})

        if r.ok:

            steps = r.json().get("steps", {})

            for step, value in steps.items():

                st.markdown(f"**{step}** → `{value}`")



use_refined = False

refined_query = query



if query and search_mode == "Enhanced":

    with st.expander("✨ Query Refinement (Enhanced Mode)", expanded=True):

        st.caption("تحسين الاستعلام: تطبيع، تصحيح إملائي، اقتراحات، Pseudo Relevance Feedback")



        c1, c2, c3 = st.columns(3)

        with c1:

            enable_spell = st.checkbox("Spell Correction", value=True)

        with c2:

            enable_suggest = st.checkbox("Suggestions", value=True)

        with c3:

            enable_prf = st.checkbox("PRF Expansion", value=True)



        if not st.session_state.get("query_history"):

            st.session_state["query_history"] = []



        if st.button("✨ Refine Query", use_container_width=True):

            with st.spinner("Refining..."):

                r = requests.post(f"{API}/refine", json={

                    "dataset": dataset,

                    "query": query,

                    "model": model,

                    "enable_spell_correct": enable_spell,

                    "enable_suggestions": enable_suggest,

                    "enable_prf": enable_prf,

                })

            if r.ok:

                st.session_state["refinement"] = r.json()

                if query not in st.session_state["query_history"]:

                    st.session_state["query_history"].insert(0, query)

                    st.session_state["query_history"] = st.session_state["query_history"][:5]

            else:

                st.error(r.text)



        ref = st.session_state.get("refinement")

        if ref and ref.get("original") == query:

            st.markdown(f"**Original:** `{ref['original']}`")

            st.markdown(f"**Refined:** `{ref['refined']}`")

            if ref.get("corrections"):

                for c in ref["corrections"]:

                    st.caption(f"✏️ `{c['original']}` → `{c['corrected']}`")

            refined_query = ref["refined"]

            use_refined = True

        elif search_mode == "Enhanced":

            st.info("اضغط Refine Query أو Search سيُطبّق التحسين تلقائيًا.")



elif query and search_mode == "Basic":

    st.caption("🔹 Basic mode: البحث مباشرة بدون Query Refinement")



search_query = refined_query if (search_mode == "Enhanced" and use_refined and refined_query) else query



if search_btn and search_query:

    if search_mode == "Enhanced" and not use_refined:

        with st.spinner("Applying query refinement..."):

            r = requests.post(f"{API}/refine", json={

                "dataset": dataset,

                "query": query,

                "model": model,

                "enable_spell_correct": True,

                "enable_suggestions": True,

                "enable_prf": True,

            })

        if r.ok:

            search_query = r.json().get("refined", query)

            st.session_state["refinement"] = r.json()



    payload = {

        "dataset": dataset,

        "query": search_query,

        "model": model,

        "top_k": top_k,

        "bm25_k1": bm25_k1,

        "bm25_b": bm25_b,

        "bm25_weight": bm25_w,

        "emb_weight": emb_w,

    }

    with st.spinner(f"Searching with {model} ({search_mode})..."):

        r = requests.post(f"{API}/search", json=payload)



    if r.ok:

        data = r.json()

        results = data.get("results", [])

        elapsed = data.get("elapsed_sec", 0)



        mode_label = "Enhanced" if search_mode == "Enhanced" else "Basic"

        st.markdown(

            f"**{len(results)} results** in `{elapsed}s` — "

            f"model: `{model}` — mode: **{mode_label}** — dataset: `{dataset}`"

        )

        if search_query != query:

            st.markdown(f"🔎 Query used: `{search_query}` (original: `{query}`)")



        info = MODEL_INFO.get(model, {})

        st.info(

            f"**Matching:** {info.get('matching', '-')}  |  "

            f"**Ranking:** {info.get('ranking', '-')}  |  "

            f"النتائج مرتّبة من الأعلى score للأقل ↓"

        )

        st.divider()



        if not results:

            st.warning("No results found.")

        else:

            for res in results:

                with st.container():

                    col_rank, col_content = st.columns([1, 10])

                    with col_rank:

                        st.markdown(f"### #{res['rank']}")

                        st.caption(f"Score: {res['score']}")

                    with col_content:

                        st.markdown(f"**`{res['doc_id']}`**")

                        st.write(res.get("snippet", "") + (" ..." if res.get("snippet") else ""))

                        with st.expander("📄 View full document"):

                            full = fetch_full_doc(dataset, res["doc_id"])

                            if full:

                                st.text_area(

                                    "Full content",

                                    full,

                                    height=300,

                                    key=f"doc_{res['doc_id']}_{res['rank']}",

                                    label_visibility="collapsed",

                                )

                                st.caption(f"Words: {len(full.split()):,} · Source: SQLite DB")

                            else:

                                st.error("Could not load full document from database.")

                    st.divider()

    else:

        st.error(f"Error: {r.text}")



# ── Index ────────────────────────────────────────────────────────────

st.markdown("---")

with st.expander("🗂️ Inverted Index"):

    col_build, col_term = st.columns(2)



    with col_build:

        st.markdown("**Index** (يُبنى تلقائيًا عند Load Dataset ويُحفظ مضغوطًا)")

        if st.button("📋 Show Index Stats", use_container_width=True):

            r = requests.get(f"{API}/index/{dataset}/stats")

            if r.ok:

                d = r.json()

                m1, m2, m3 = st.columns(3)

                m1.metric("Docs indexed", f"{d['num_docs']:,}")

                m2.metric("Unique terms", f"{d['num_terms']:,}")

                m3.metric("Avg doc length", f"{d['avg_doc_length']}")

                if d.get("on_disk"):

                    st.success("Index persisted on disk ✓")

                st.code("  |  ".join(d["top_10_terms"]))

            else:

                st.error("Index not built yet. Load dataset first.")



        if st.button("🔄 Rebuild Index", use_container_width=True):

            with st.spinner("Rebuilding..."):

                r = requests.post(f"{API}/index/{dataset}")

            if r.ok:

                d = r.json()

                st.success(f"✅ Index rebuilt in {d['elapsed_sec']}s")

            else:

                st.error(r.text)



    with col_term:

        st.markdown("**Term Lookup**")

        lookup_term = st.text_input("Search a term in the index", placeholder="e.g. water")

        if st.button("🔍 Lookup Term") and lookup_term:

            r = requests.get(f"{API}/index/{dataset}/term/{lookup_term}")

            if r.ok:

                d = r.json()

                st.info(f"Term: **{d['term']}** — found in **{d['document_frequency']}** documents")

                st.json(d["postings_sample"])

            else:

                st.error(r.text)



# ── Evaluation ───────────────────────────────────────────────────────

st.markdown("---")

with st.expander("📊 Evaluation & Comparison"):

    eval_tab1, eval_tab2 = st.tabs(["Single Model", "All Models Comparison"])



    with eval_tab1:

        eval_model = st.selectbox("Model to evaluate", ALL_MODELS, key="eval_model")

        eval_topk = st.slider("Top-K", 5, 50, 10, key="eval_topk")



        if st.button("▶ Run Evaluation", key="eval_single"):

            with st.spinner("Evaluating..."):

                r = requests.get(

                    f"{API}/evaluate",

                    params={"dataset": dataset, "model": eval_model, "top_k": eval_topk},

                    timeout=36000,

                )

            if r.ok:

                m = r.json().get("metrics", {})

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("MAP", f"{m.get('map', 0):.4f}")

                c2.metric("Recall", f"{m.get('recall', 0):.4f}")

                c3.metric("Precision@10", f"{m.get('precision_at_10', 0):.4f}")

                c4.metric("nDCG", f"{m.get('ndcg', 0):.4f}")

                st.caption(f"Queries evaluated: {int(m.get('num_queries', 0))}")

            else:

                st.error(r.text)



    with eval_tab2:

        eval_topk_all = st.slider("Top-K (all models)", 5, 50, 10, key="eval_topk_all")

        if st.button("▶ Evaluate All Models", key="eval_all"):

            with st.spinner("Evaluating all 5 models — may take a while..."):

                r = requests.get(

                    f"{API}/evaluate/all",

                    params={"dataset": dataset, "top_k": eval_topk_all},

                    timeout=36000,

                )

            if r.ok:

                rows = r.json().get("results", [])

                st.session_state["eval_comparison"] = rows

            else:

                st.error(r.text)



        if st.session_state.get("eval_comparison"):

            rows = st.session_state["eval_comparison"]

            df = pd.DataFrame(rows)

            if "error" not in df.columns:

                st.dataframe(df.set_index("model"), use_container_width=True)



                metric_cols = ["map", "recall", "precision_at_10", "ndcg"]

                chart_df = df.set_index("model")[metric_cols]

                st.bar_chart(chart_df)

                st.line_chart(chart_df.T)

            else:

                st.dataframe(df, use_container_width=True)


# ── Extra Feature 1: Vector Store (FAISS) ────────────────────────────
st.markdown("---")
with st.expander("🧬 Vector Store (FAISS) — ميزة إضافية"):
    st.caption(
        "فهرس FAISS يخزّن الـ embeddings للبحث الدلالي السريع. "
        "يُبنى تلقائيًا عند Load Dataset (يعيد استخدام الـ embeddings المحسوبة)."
    )
    if st.button("📊 Show Vector Store Stats", use_container_width=True):
        r = requests.get(f"{API}/vector-store/{dataset}/stats")
        if r.ok:
            d = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Vectors", f"{d['num_vectors']:,}")
            c2.metric("Dimension", d["dimension"])
            c3.metric("Backend", d["backend"])
            st.success(f"Index type: {d['index_type']}")
        else:
            st.error(r.text)

    if st.button("🔄 Rebuild Vector Store", use_container_width=True):
        with st.spinner("Building FAISS index..."):
            r = requests.post(f"{API}/vector-store/{dataset}", params={"rebuild": True}, timeout=36000)
        if r.ok:
            st.success("✅ Vector store rebuilt")
        else:
            st.error(r.text)

    st.info("للبحث عبر الـ Vector Store: اختر النموذج `vector_store` من الـ sidebar.")


# ── Extra Feature 2: Documents Clustering ────────────────────────────
st.markdown("---")
with st.expander("🗃️ Documents Clustering — ميزة إضافية"):
    st.caption(
        "تجميع الوثائق المتشابهة دلاليًا في عناقيد (KMeans على الـ embeddings). "
        "كل عنقود يظهر أهم كلماته وعدد وثائقه."
    )
    n_clusters = st.slider("عدد العناقيد (K)", 2, 20, 8, key="n_clusters")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("▶ Run Clustering", use_container_width=True):
            with st.spinner("Clustering documents..."):
                r = requests.post(
                    f"{API}/cluster/{dataset}",
                    params={"n_clusters": n_clusters, "rebuild": True},
                    timeout=36000,
                )
            if r.ok:
                st.session_state["clusters"] = r.json()
            else:
                st.error(r.text)
    with col_c2:
        if st.button("📂 Load Saved Clusters", use_container_width=True):
            r = requests.get(f"{API}/cluster/{dataset}")
            if r.ok:
                st.session_state["clusters"] = r.json()
            else:
                st.error(r.text)

    clusters_data = st.session_state.get("clusters")
    if clusters_data and clusters_data.get("clusters"):
        st.markdown(
            f"**{clusters_data['n_clusters']} clusters** على "
            f"**{clusters_data['total_docs']:,}** وثيقة "
            f"(inertia: {clusters_data.get('inertia', '-')})"
        )

        sizes_df = pd.DataFrame(
            [{"cluster": c["label"], "size": c["size"]} for c in clusters_data["clusters"]]
        ).set_index("cluster")
        st.bar_chart(sizes_df)

        for c in clusters_data["clusters"]:
            with st.container():
                st.markdown(f"**Cluster {c['cluster_id']}** — {c['size']:,} docs")
                st.markdown("الكلمات المميّزة: " + "  ".join(f"`{t}`" for t in c["top_terms"]))
                with st.expander(f"عيّنة وثائق (Cluster {c['cluster_id']})"):
                    for doc_id in c["sample_doc_ids"]:
                        full = fetch_full_doc(dataset, doc_id)
                        st.markdown(f"**`{doc_id}`**")
                        st.caption((full or "")[:200] + " ...")
                st.divider()


