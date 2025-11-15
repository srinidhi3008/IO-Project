# IO-Project
# AI-Powered Email Reply API

This project is a Flask API server that uses LLMs (`microsoft/phi-2`) and a vector database (`FAISS`) to analyze emails and generate intelligent, context-aware replies. It is designed to be the "brain" for any email client, such as a browser extension, desktop app, or mobile app.

---

## Tech Stack

* **Framework:** Flask
* **AI/ML:** `transformers` (Hugging Face)
* **Generator Model:** `microsoft/phi-2` (text generation)
* **Tone Model:** `sentiment-analysis` pipeline
* **RAG (Retrieval):** `sentence-transformers` (embeddings) & `faiss-cpu` (vector search)
* **Dependencies:** `torch`, `numpy`, `sentencepiece`

---

## API Endpoints

### Core AI

* `POST /generate_reply`
    * **Body:** `{"sender": "...", "email_text": "..."}`
    * **Action:** The main "Agent" endpoint. It automatically detects the tone (or uses a sender rule) and generates a single, context-aware reply using RAG.
    * **Returns:** The new draft reply object.

* `POST /parse_email`
    * **Body:** `{"email_text": "..."}`
    * **Action:** Uses `phi-2` to summarize the email into key action items and questions.
    * **Returns:** `{"main_points": "..."}`

* `POST /reply_variants`
    * **Body:** `{"email_text": "...", "tone": "(optional)"}`
    * **Action:** Generates 2-3 different reply options for the given email.
    * **Returns:** `{"tone": "...", "variants": [...]}`

### Memory & Management

* `POST /set_tone`
    * **Body:** `{"sender": "...", "tone": "formal" | "friendly" | "neutral"}`
    * **Action:** "Trains" the agent by creating a rule to always use a specific tone for a specific sender.
    * **Returns:** A success message.

* `POST /add_signature`
    * **Body:** `{"signature": "Best,\nMy Name"}`
    * **Action:** Sets or updates the global signature that will be attached to approved replies.
    * **Returns:** A success message.

* `POST /approve_reply`
    * **Body:** `{"reply_id": 1}`
    * **Action:** Finalizes a draft. It finds the draft by its ID, adds the saved signature, and moves it to the permanent history.
    * **Returns:** The final, approved reply object (with `final_text`).

* `GET /saved_replies`
    * **Action:** View all drafts that are pending approval.
    * **Returns:** A list of reply drafts.

* `GET /reply_history/`
    * **Action:** View all replies that have been approved.
    * **Returns:** A list of approved replies.
```eof
