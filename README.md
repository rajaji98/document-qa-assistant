# Document Q&A Assistant

A Streamlit application for asking questions about uploaded PDF/TXT documents using Sentence Transformers embeddings, FAISS semantic search, BM25 hybrid retrieval, Gemini answer generation, source citations, chat history, and a retrieval evaluation dashboard.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with:

   ```dotenv
   GEMINI_API_KEY=your_actual_key
   GEMINI_MODEL=your_supported_gemini_model
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. Open https://share.streamlit.io/ and create a new app.
3. Select the repository, branch, and `app.py` as the main file.
4. In the app's Settings / Secrets section, add:

   ```toml
   GEMINI_API_KEY = "your_actual_key"
   GEMINI_MODEL = "your_supported_gemini_model"
   ```

5. Save secrets and redeploy/reboot the app if needed.

Do not commit `.env`, actual API keys, private documents, or `secrets.toml`.

## Notes

- The app reads GEMINI_API_KEY and GEMINI_MODEL from local environment/.env during development and from Streamlit Community Cloud Secrets after deployment. The app needs internet access to call Gemini and download/load the embedding model on its first run.
- Initial startup may be slow because model weights are downloaded and cached at runtime.
- Uploaded documents are processed by the running app; do not upload sensitive documents to a public demo unless you have permission and understand the hosting/privacy implications.
- `requirements.txt` uses unpinned package versions. If a dependency compatibility issue occurs in the cloud, pin the known-working versions from your local environment.
