"""
🎬 TMDB AI System — Production Grade (Final Improved Version)
"""

import os
import numpy as np
import pandas as pd
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from sklearn.ensemble import GradientBoostingRegressor

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")

DATA_DIR = "data"
CREDITS_PATH = f"{DATA_DIR}/tmdb_5000_credits.csv"
MOVIES_PATH  = f"{DATA_DIR}/tmdb_5000_movies.csv"

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ─────────────────────────────
# STATE
# ─────────────────────────────
state = {"ready": False}

# ─────────────────────────────
# PIPELINE
# ─────────────────────────────
def build_pipeline():
    logs = []

    if not os.path.exists(CREDITS_PATH) or not os.path.exists(MOVIES_PATH):
        return "❌ Dataset not found in /data"

    logs.append("📂 Loading dataset...")

    credits = pd.read_csv(CREDITS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    df = pd.merge(movies, credits, left_on="id", right_on="movie_id")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.fillna(2000)

    # ── SAFE RAG DATA ──
    df = df.dropna(subset=["overview", "budget", "revenue"]).reset_index(drop=True)
    df_rag = df[df["budget"] > 0].reset_index(drop=True)

    # ── TEXT ──
    df_rag["text"] = df_rag.apply(
        lambda r: f"{r['title']} | {str(r['overview'])[:300]} | Budget {r['budget']} | Revenue {r['revenue']}",
        axis=1
    )

    # ── EMBEDDINGS ──
    embedder = SentenceTransformer(EMBED_MODEL)
    emb = embedder.encode(df_rag["text"].tolist(), normalize_embeddings=True)
    emb = np.array(emb).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # ── ML MODEL ──
    features = ["budget", "vote_average", "vote_count", "release_year"]

    df_rag = df_rag.dropna(subset=features + ["revenue"])

    X = df_rag[features]
    y = df_rag["revenue"]

    model = GradientBoostingRegressor(n_estimators=250, random_state=42)
    model.fit(X, y)

    # ── LLM ──
    llm = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

    # ── STATE ──
    state.update({
        "df": df_rag,
        "index": index,
        "embedder": embedder,
        "model": model,
        "features": features,
        "llm": llm,
        "ready": True
    })

    logs.append("✅ System Ready (Production Mode)")
    return "\n".join(logs)

# ─────────────────────────────
# GUARD
# ─────────────────────────────
def guard():
    if not state["ready"]:
        return False, "⚠️ System not ready — run pipeline first"
    return True, None

# ─────────────────────────────
# SEARCH (RAG)
# ─────────────────────────────
def search(query):
    ok, msg = guard()
    if not ok:
        return msg

    q = state["embedder"].encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = state["index"].search(q, TOP_K)

    results = []
    for i, score in zip(idx[0], scores[0]):
        row = state["df"].iloc[i]
        results.append(
            f"""🎬 {row['title']}
💰 Revenue: ${row['revenue']:,.0f}
📊 Similarity: {score:.3f}
📅 Year: {row['release_year']}
"""
        )

    return "\n---\n".join(results)

# ─────────────────────────────
# PREDICT
# ─────────────────────────────
def predict(budget, vote_avg, vote_count, year):
    ok, msg = guard()
    if not ok:
        return msg

    model = state["model"]
    features = state["features"]

    x = pd.DataFrame([[budget, vote_avg, vote_count, year]], columns=features)

    pred = model.predict(x)[0]

    return f"""
🎯 Predicted Revenue: ${pred:,.0f}
📊 ${pred/1e6:.2f}M
📈 ROI Estimate: {pred/max(budget,1):.2f}x
"""

# ─────────────────────────────
# AGENT
# ─────────────────────────────
def agent(question):
    if not state["llm"]:
        return "⚠️ HF_TOKEN not set"

    resp = state["llm"].chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=400
    )

    return resp.choices[0].message.content

# ─────────────────────────────
# INIT
# ─────────────────────────────
def init():
    return build_pipeline()

# ─────────────────────────────
# UI
# ─────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🎬 TMDB AI System — Production Grade")

    status = gr.Textbox(label="System Status")

    with gr.Tab("🔍 Search (RAG)"):
        q = gr.Textbox(label="Query")
        o = gr.Textbox()
        gr.Button("Search").click(search, q, o)

    with gr.Tab("💰 Predict"):
        b = gr.Slider(1, 500, label="Budget")
        v = gr.Slider(0, 10, label="Rating")
        vc = gr.Slider(0, 10000, label="Votes")
        y = gr.Slider(1990, 2030, label="Year")

        o = gr.Textbox()
        gr.Button("Predict").click(predict, [b, v, vc, y], o)

    with gr.Tab("🤖 Agent"):
        q2 = gr.Textbox(label="Ask AI")
        o2 = gr.Textbox()
        gr.Button("Ask").click(agent, q2, o2)

    app.load(init, outputs=status)

# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    app.launch(share=True)