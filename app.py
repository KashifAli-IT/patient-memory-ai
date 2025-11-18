# app.py
import os
import uuid
from typing import List, Dict
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import gradio as gr

# Load env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "patient_memory"

# Initialize embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast

# Init Qdrant
if not QDRANT_URL or not QDRANT_API_KEY:
    raise EnvironmentError("Set QDRANT_URL and QDRANT_API_KEY in your environment or .env file")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )

# --- Ensure collection always has demo data ---
# Check if collection is empty
existing_points = client.scroll(collection_name=COLLECTION_NAME, limit=1)
if len(existing_points) == 0:
    # Collection empty, add demo notes automatically
    prepopulate_demo_data()

def embed_text(text: str) -> List[float]:
    vec = embedder.encode([text], show_progress_bar=False)[0]
    return vec.tolist()

def store_patient_note(patient_id: str, text: str, metadata: Dict = None):
    """Store a patient note in Qdrant under a given patient_id."""
    if metadata is None:
        metadata = {}
    vec = embed_text(text)
    point_id = str(uuid.uuid4())
    payload = {"patient_id": patient_id, "text": text}
    payload.update(metadata)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": vec,
            "payload": payload
        }]
    )
    return point_id

def retrieve_similar(patient_id: str, query: str, top_k: int = 5):
    qv = embed_text(query)
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qv,
        limit=top_k,
        query_filter={"must": [{"key": "patient_id", "match": {"value": patient_id}}]}
    )
    results = []
    for h in hits:
        results.append({"score": float(h.score), "id": h.id, "text": h.payload.get("text")})
    return results

def summarize_contexts(contexts: List[Dict]):
    """Very small rule-based summarizer: join contexts and extract lines."""
    # For quick demo: just concatenate the top contexts and return as 'memory summary'
    texts = [c["text"] for c in contexts]
    joined = "\n\n".join(texts)
    # Optionally limit length
    if len(joined) > 2000:
        joined = joined[:2000] + "..."
    return joined

def answer_patient_question(patient_id: str, question: str):
    # Retrieve memory
    contexts = retrieve_similar(patient_id, question, top_k=5)
    if not contexts:
        return "No prior history found for this patient. Consider adding notes first."

    memory_summary = summarize_contexts(contexts)

    # Simple template-based answer using memory (no external LLM)
    # SMART: for hackathon demo you can later swap this with Gemini/OpenAI call.
    answer = f"Found {len(contexts)} relevant past notes (top results shown):\n\n"
    for i,c in enumerate(contexts, start=1):
        answer += f"{i}. {c['text'][:200]} (score: {c['score']:.3f})\n\n"

    answer += "\n=== Suggested follow-ups / insights (auto) ===\n"
    # very simple heuristics to offer follow-ups:
    if "allergy" in memory_summary.lower():
        answer += "- Patient has recorded allergies — ensure medications avoid known allergens.\n"
    if "diabetes" in memory_summary.lower() or "blood sugar" in memory_summary.lower():
        answer += "- History suggests diabetes-related notes — consider blood sugar monitoring.\n"
    if "fever" in memory_summary.lower() or "cough" in memory_summary.lower():
        answer += "- Recent fever or cough mentioned — check vitals and recent symptom timeline.\n"

    answer += "- For accurate clinical advice, cross-check with a physician. This tool is for memory/context only."

    return answer

# Sample prepopulate function (creates demo patient + notes)
def prepopulate_demo_data():
    pid = "patient_001"
    notes = [
        "Patient: John Doe, 35M. Type 2 diabetes, on metformin 500mg twice daily. Allergic to penicillin.",
        "Visit 2025-10-01: Complains of mild cough, no fever. BP 120/80. Advised hydration and rest.",
        "Visit 2025-11-05: Reports elevated blood sugar after dietary change. Suggested diet plan.",
    ]
    for n in notes:
        store_patient_note(pid, n, metadata={"demo": True})
    return pid

# Build Gradio interface
with gr.Blocks(theme=None) as demo:
    gr.Markdown("# Patient Memory — Qdrant Demo\n"
                "Store patient notes and ask questions. This demo uses Qdrant Cloud and local sentence-transformers.")
    with gr.Row():
        with gr.Column(scale=1):
            patient_id_in = gr.Textbox(label="Patient ID", value="patient_001", interactive=True)
            note_in = gr.Textbox(label="Add patient note (free text)", lines=4, placeholder="e.g. Visit: ...")
            add_btn = gr.Button("Store Note")
            pre_btn = gr.Button("Prepopulate demo data")
            add_status = gr.Textbox(label="Store status", interactive=False)
            history_btn = gr.Button("List recent notes")
            history_out = gr.Dataframe(headers=["id", "score", "text"], interactive=False)
        with gr.Column(scale=1):
            question_in = gr.Textbox(label="Ask a question about patient", lines=3, placeholder="e.g. Does patient have allergies?")
            ask_btn = gr.Button("Ask")
            answer_out = gr.Textbox(label="Answer", lines=20)

    def _add_note(patient_id, note):
        if not note.strip():
            return "Please provide a note to store."
        pid = patient_id.strip() or "patient_001"
        pid = pid.replace(" ", "_")
        store_patient_note(pid, note)
        return f"Stored note for {pid}."

    def _list_history(patient_id):
        pid = (patient_id.strip() or "patient_001").replace(" ", "_")
        # quick search wildcard: retrieve top 20 entries for patient by searching a neutral query
        res = client.search(collection_name=COLLECTION_NAME, query_vector=embed_text("patient"), limit=50,
                            query_filter={"must":[{"key":"patient_id","match":{"value":pid}}]})
        rows = []
        for r in res:
            rows.append([r.id, float(r.score), r.payload.get("text")])
        if not rows:
            return pd.DataFrame([], columns=["id","score","text"])
        return pd.DataFrame(rows, columns=["id","score","text"])

    def _prepopulate():
        pid = prepopulate_demo_data()
        return f"Prepopulated demo data for {pid}."

    def _ask(patient_id, question):
        pid = (patient_id.strip() or "patient_001").replace(" ", "_")
        if not question.strip():
            return "Please type a question to ask."
        return answer_patient_question(pid, question)

    add_btn.click(_add_note, inputs=[patient_id_in, note_in], outputs=[add_status])
    pre_btn.click(_prepopulate, inputs=[], outputs=[add_status])
    history_btn.click(_list_history, inputs=[patient_id_in], outputs=[history_out])
    ask_btn.click(_ask, inputs=[patient_id_in, question_in], outputs=[answer_out])

if __name__ == "__main__":
    demo.launch()
