# RAG Pipeline Demo

Quick-start:

1. Copy `.env` to the project root and fill in your credentials (Databricks token, host, endpoint) if you plan to use the model serving layer.

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Prepare documents:
- Put PDF, DOCX, TXT, CSV, XLS(X) files into the `data/` folder or provide `SPECIFIC_FILES` in `.env`.

4. Run the Streamlit UI:

```bash
streamlit run src/app.py
```

Notes:
- `src/config.py` reads configuration and environment variables from `.env`.
- If Databricks credentials are missing or invalid, the code uses a simple fallback that returns the document context instead of a model-generated answer.
- The project prefers local cached `sentence-transformers` models when available; otherwise it falls back to a TF-IDF embedding approach.

If you want me to pin exact package versions, run automated checks, or add a quick `main.py` runner, tell me and I will add them.
