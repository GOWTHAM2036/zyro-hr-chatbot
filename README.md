# Zyro Dynamics HR Assistant

A Streamlit-based HR chatbot that answers employee questions using Zyro Dynamics policy PDFs with a Retrieval-Augmented Generation (RAG) pipeline.

## What this project does

- Loads HR policy documents from `pdfs/`
- Splits documents into chunks
- Creates FAISS vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Retrieves relevant context for each user question
- Uses Groq (`llama-3.3-70b-versatile`) to generate responses grounded in the policy content

## Repository structure

- `app.py` – Streamlit app and RAG chain setup
- `requirements.txt` – Python dependencies
- `pdfs/` – HR policy source documents

## Requirements

- Python 3.11+
- A Groq API key

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your Streamlit secret in `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

## Run the app

```bash
streamlit run app.py
```

Then open the Streamlit URL (typically `http://localhost:8501`) and ask HR-related questions.

## Notes

- The app only indexes `.pdf` files in the `pdfs/` directory.
- If no relevant context is found, it returns a policy-limited fallback response.
