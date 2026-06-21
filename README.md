# langchain-rag

This project demonstrates building a Retrieval-Augmented Generation (RAG) system for API documentation assistance. It includes two hands-on labs:

- **Lab 1**: Jupyter notebook (`Handson_lab1.ipynb`) for setting up and testing the RAG pipeline (uses Groq for LLM and OpenAI for embeddings).
- **Lab 2**: Streamlit web app (`streamlit_app.py`) providing an interactive interface for querying API docs, running agents, and viewing traces (uses Groq for LLM and OpenAI for embeddings).

## Prerequisites
- Python 3.8+
- For Lab 1: Groq API key and OpenAI API key
- For Lab 2: Groq API key and OpenAI API key

### Install Ollama (for Lab 1)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Pull required models:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Tech Stack
- **Python**: Core language
- **LangChain**: Framework for LLM applications
- **Chroma**: Vector database for embeddings
- **Ollama** (Lab 1): Local LLM and embedding models
- **Groq** (Labs 1 & 2): Cloud LLM
- **OpenAI** (Labs 1 & 2): Embeddings
- **Streamlit**: Web app framework
- **LangGraph**: For agent memory and orchestration

## How to Run

### Lab 1: RAG Pipeline Setup
1. Set environment variables:
   ```bash
   export LANGSMITH_TRACING=true
   export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   export LANGSMITH_API_KEY="your-langsmith-key"
   export LANGSMITH_PROJECT=pr-loyal-egghead-68
   export GROQ_API_KEY="your-groq-api-key"
   ```
2. Open `Handson_lab1.ipynb` in Jupyter.
3. Run cells sequentially to:
   - Install dependencies
   - Load and chunk documents
   - Create vector store
   - Test retrieval and RAG chain
   - Set up agent with tools

### Lab 2: Streamlit API Assistant
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables:
   ```bash
   export LANGSMITH_TRACING=true
   export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   export LANGSMITH_API_KEY=""
   export LANGSMITH_PROJECT=pr-loyal-egghead-68
   export GROQ_API_KEY="your-groq-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   ```
3. Place API documentation files (.md, .pdf) in `./api_docs/`
4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Access the web interface at `http://localhost:8501`

### Deployment on Streamlit Cloud
1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo.
3. Set the main file to `streamlit_app.py`.
4. Add `GROQ_API_KEY` and `OPENAI_API_KEY` in the secrets.
5. Deploy.

### Docker Deployment
1. Build the image:
   ```bash
   docker build -t rag-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 -e GROQ_API_KEY="your-groq-key" -e OPENAI_API_KEY="your-openai-key" rag-app
   ```

## Features
- Document loading and chunking
- Vector embeddings and retrieval
- RAG-based Q&A
- Agent with tools (calculator, doc search)
- Guardrails for query safety
- Retrieval evaluation metrics
- LangSmith integration for tracing
