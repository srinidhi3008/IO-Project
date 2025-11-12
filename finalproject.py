from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import torch
import os
import time 
app = Flask(__name__)

db = {
    "drafts": {},
    "reply_logs": [],
    "settings": {"signature": "Best regards,\nYour Name"},
    "sender_rules": {
        "ceo@company.com": "formal",
        "friend@gmail.com": "friendly"
    }
}
draft_counter = 0


GENERATOR_MODEL = "microsoft/phi-2"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_FILE = "knowledge_base.txt"

print("Loading models... This may take a moment.")
try:
    generator = pipeline(
        "text-generation", 
        model=GENERATOR_MODEL, 
        device=0 if torch.cuda.is_available() else -1,
        trust_remote_code=True
    )
    
    sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    print("All models loaded successfully.")

except Exception as e:
    print(f"[FATAL] Could not load models: {e}")
    generator = None
    sentiment_analyzer = None
    embedder = None

try:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    
    if not docs:
        print(f"[WARN] {KNOWLEDGE_FILE} is empty. Retrieval will be disabled.")
        index = None
    else:
        print(f"Embedding {len(docs)} documents from {KNOWLEDGE_FILE}...")
        doc_embeddings = embedder.encode(docs)
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(np.array(doc_embeddings))
        print("FAISS index built.")

except FileNotFoundError:
    print(f"[WARN] {KNOWLEDGE_FILE} not found. Retrieval will be disabled.")
    docs = []
    index = None

TONE_LABELS = ["formal", "friendly", "neutral"]

def retrieve_context(email_text, top_k=3):
    if index is None:
        return ""
    try:
        query_embedding = embedder.encode([email_text])
        distances, indices = index.search(np.array(query_embedding), top_k)
        retrieved = [docs[i] for i in indices[0]]
        return "\nHere is some relevant context:\n" + "\n".join(retrieved)
    except Exception as e:
        print(f"[WARN] FAISS retrieval failed: {e}")
        return ""

def detect_tone(email_text):
    if not sentiment_analyzer:
        return "neutral"
    try:
        result = sentiment_analyzer(email_text)
        sentiment = result[0]['label']
        if sentiment == 'POSITIVE':
            return "friendly"
        elif sentiment == 'NEGATIVE':
            return "formal" 
        else:
            return "neutral"
    except Exception as e:
        print(f"[WARN] Sentiment analysis failed: {e}")
        return "neutral"

def _run_phi_generation(prompt, params):
    if not generator:
        return ["Error: Generator model not loaded."]
    try:
        print("[DEBUG] generation prompt:", prompt[:300].replace("\n", " "))
        print("[DEBUG] generation params:", params)
        results = generator(prompt, **params)
        print("[DEBUG] raw results:", results)
        normalized = []
        if isinstance(results, dict):
            results = [results]

        for res in results:
            if isinstance(res, dict) and "generated_text" in res:
                text = res["generated_text"]
            else:
                text = str(res)

        
            reply = re.split(r'Reply:', text, maxsplit=1)[-1].strip()
            if not reply:
                reply = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
            normalized.append(reply)

        print("[DEBUG] normalized replies:", normalized)
        return normalized

    except Exception as e:
        print(f"[FATAL] Generation failed: {e}")
        import traceback; traceback.print_exc()
        return [f"Error: Could not generate reply. {e}"]

def generate_reply(email_text, tone="neutral", signature=None):
    context = retrieve_context(email_text)
    prompt = (
        f"You are an email assistant. Write a {tone} reply to this email. "
        f"Use the context provided if it is relevant.\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Email:**\n{email_text}\n\n"
        f"Reply:"
    )
    
    params = {
        "max_new_tokens": 150,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True
    }
    
    reply = _run_phi_generation(prompt, params)[0]
    
    if signature:
        reply += f"\n\n{signature}"
    return reply

def extract_main_points(email_text: str) -> str:
    prompt = (
        f"You are a summarization assistant. Summarize the key action items and questions from the following email. "
        f"Present them as a bulleted list.\n\n"
        f"**Email:**\n{email_text}\n\n"
        f"Reply:" 
    )
    params = {
        "max_new_tokens": 100,
        "num_return_sequences": 1,
        "temperature": 0.3, 
        "do_sample": True
    }
    return _run_phi_generation(prompt, params)[0]

def generate_reply_variants(email_text: str, tone: str) -> list:
    context = retrieve_context(email_text)
    prompt = (
        f"You are an email assistant. Write a {tone} reply to this email. "
        f"Use the context provided if it is relevant.\n\n"
        f"*Context:*\n{context}\n\n"
        f"*Email:*\n{email_text}\n\n"
        f"Reply:"
    )

    params = {
        "max_new_tokens": 150,
        "num_return_sequences": 2,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }

    return _run_phi_generation(prompt, params)


# 1. POST /parse_email
@app.route("/parse_email", methods=["POST"])
def parse_email_api():
    data = request.get_json(force=True)
    text = data.get("email_text", "").strip()
    if not text:
        return jsonify({"error": "email_text is required"}), 400

    points = extract_main_points(text)
    return jsonify({"main_points": points})

# 2. POST /generate_reply (with Agent)
@app.route("/generate_reply", methods=["POST"])
def generate_reply_api():
    global draft_counter
    data = request.get_json(force=True)
    text = data.get("email_text", "").strip()
    sender = data.get("sender", "").strip()
    
    if not text:
        return jsonify({"error": "email_text is required"}), 400


    tone = db["sender_rules"].get(sender)
    if not tone:
        print(f"[Agent] No rule for {sender}. Detecting tone...")
        tone = detect_tone(text)
    else:
        print(f"[Agent] Found rule for {sender}. Using tone: {tone}")
        
    reply = generate_reply(text, tone) 
    
    draft_counter += 1
    draft_id = draft_counter
    draft = {
        "id": draft_id, "sender": sender, "email_text": text,
        "reply_text": reply, "tone": tone, "status": "draft",
        "created_at": time.time()
    }
    db["drafts"][draft_id] = draft
    
    return jsonify(draft), 201

# 3. POST /set_tone (Agent Training)
@app.route("/set_tone", methods=["POST"])
def set_tone_api():
    data = request.get_json(force=True)
    sender = data.get("sender", "").strip()
    tone = data.get("tone", "").strip()
    
    if not sender or not tone:
        return jsonify({"error": "sender and tone are required"}), 400
        
    if tone not in TONE_LABELS:
        return jsonify({"error": f"Invalid tone. Must be one of: {TONE_LABELS}"}), 400
    
    db["sender_rules"][sender] = tone
    return jsonify({"message": f"Rule set: Sender '{sender}' will now use tone '{tone}'."})

# 4. GET /saved_replies
@app.route("/saved_replies", methods=["GET"])
def get_saved_replies():
    return jsonify(list(db["drafts"].values()))

# 5. POST /approve_reply
@app.route("/approve_reply", methods=["POST"])
def approve_reply_api():
    data = request.get_json(force=True)
    try:
        reply_id = int(data.get("reply_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "reply_id (integer) is required"}), 400
        
    draft = db["drafts"].pop(reply_id, None)
    
    if not draft:
        return jsonify({"error": f"Draft reply with id {reply_id} not found."}), 404
        
    signature = db["settings"]["signature"]
    draft["final_text"] = f"{draft['reply_text']}\n\n{signature}"
    draft["status"] = "approved"
    draft["approved_at"] = time.time()
    
    db["reply_logs"].append(draft)
    return jsonify(draft)

# 6. POST /reply_variants
@app.route("/reply_variants", methods=["POST"])
def reply_variants_api():
    data = request.get_json(force=True)
    text = data.get("email_text", "").strip()
    tone = data.get("tone", None)
    print("[API] /reply_variants called with text:", text[:200], "tone:", tone)
    if not text:
        return jsonify({"error": "email_text is required"}), 400
    if not tone:
        tone = detect_tone(text)
        print("[API] detected tone:", tone)
    variants = generate_reply_variants(text, tone)
    print("[API] generated variants:", variants)
    return jsonify({"tone": tone, "variants": variants})
# 7. POST /add_signature
@app.route("/add_signature", methods=["POST"])
def add_signature_api():
    data = request.get_json(force=True)
    signature = data.get("signature")
    if signature is None:
        return jsonify({"error": "signature is required"}), 400
        
    db["settings"]["signature"] = signature
    return jsonify({"message": "Signature updated successfully.", "signature": signature})

# 8. GET /reply_history/
@app.route("/reply_history/", methods=["GET"])
def get_reply_history():
    return jsonify(db["reply_logs"])


if __name__ == "__main__":
    if not generator or not embedder or not sentiment_analyzer:
        print("\n[ERROR] One or more models failed to load. Exiting.")
    else:
        print("\nAll models loaded. Starting Flask server...")

        app.run(debug=True, port=5001)

