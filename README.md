# 🔬 AI/ML Research Navigator

An end-to-end Retrieval-Augmented Generation (RAG) application for asking questions about computer vision and machine learning research papers.

The application combines:

- PDF document ingestion
- Text chunking
- Hugging Face sentence embeddings
- Chroma vector database
- Similarity-based document retrieval
- Relevance filtering
- Local LLM inference using Ollama
- Qwen 2.5 3B
- FastAPI backend
- Streamlit frontend
- Answer-quality evaluation
- Out-of-domain question handling

The system is designed to provide grounded answers using only the research papers stored in its local knowledge base.

---

# Project Overview

AI/ML Research Navigator allows users to ask natural-language questions about research papers.
<img width="1909" height="870" alt="image" src="https://github.com/user-attachments/assets/246eff97-2942-4957-b116-8ee3d67793de" />


Instead of sending the question directly to an LLM, the application follows a Retrieval-Augmented Generation workflow:

```text
User Question
      ↓
FastAPI API
      ↓
Question Embedding
      ↓
Chroma Vector Database
      ↓
Similarity Search
      ↓
Relevance Filtering
      ↓
Relevant Research Paper Chunks
      ↓
Context Construction
      ↓
Qwen 2.5 3B via Ollama
      ↓
Grounded Answer
      ↓
FastAPI Response
      ↓
Streamlit UI
```

The application uses a local LLM through Ollama, so the question-answering process does not require sending the research-paper content to a cloud LLM provider.

---

# Key Features

## 1. PDF Research Paper Ingestion

Research papers are stored locally in:

```text
data/papers/
```

The ingestion pipeline:

1. Finds PDF files.
2. Loads the PDF documents.
3. Extracts text from the pages.
4. Splits the text into smaller chunks.
5. Generates embeddings for the chunks.
6. Stores the embeddings in Chroma.

The current knowledge base contains three research papers:

```text
resnet.pdf
vit.pdf
swin_transformer.pdf
```

---

## 2. Semantic Search

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert questions and document chunks into vector representations.

Chroma then performs similarity search to retrieve the most relevant chunks for the user's question.

The system retrieves up to:

```text
TOP_K = 6
```

candidate chunks.

---

## 3. Relevance Filtering

Retrieved documents are filtered using a distance threshold.

The current configuration uses:

```text
DISTANCE_THRESHOLD = 1.15
```

Only documents satisfying the configured relevance condition are passed to the answer-generation stage.

This prevents clearly irrelevant retrieved chunks from being included in the LLM context.

---

## 4. Grounded Answer Generation

The application uses:

```text
Qwen 2.5 3B
```

through:

```text
Ollama
```

The LLM is explicitly instructed to answer using only the retrieved research-paper context.

The prompt also instructs the model to:

- avoid unsupported claims
- avoid using general knowledge
- preserve technical terminology
- preserve mathematical definitions
- answer comparison questions explicitly
- avoid unnecessary information
- refuse questions when the context is insufficient

For unsupported questions, the application uses the following response:

```text
The available research papers do not contain enough information to answer this question.
```

---

# Current Knowledge Base

The current project contains three research papers.

## ResNet

```text
resnet.pdf
```

Used for questions related to:

- residual learning
- residual functions
- shortcut connections
- identity mappings
- projection shortcuts
- ResNet architecture

## Vision Transformer

```text
vit.pdf
```

Used for questions related to:

- Vision Transformer
- image patches
- patch embeddings
- Transformer encoder
- self-attention
- image representation

## Swin Transformer

```text
swin_transformer.pdf
```

Used for questions related to:

- shifted windows
- window-based attention
- cross-window connections
- Swin Transformer architecture

---

# Project Architecture

```text
research-rag/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── rag.py
│   ├── main.py
│   │
│   └── ...
│
├── data/
│   └── papers/
│       ├── resnet.pdf
│       ├── vit.pdf
│       └── swin_transformer.pdf
│
├── chroma_db/
│
├── evaluation/
│   └── evaluate_answers.py
│
├── tests/
│   └── ...
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

The exact number of application modules may vary as the project evolves.

---

# Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Qwen 2.5 3B |
| LLM Runtime | Ollama |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | Chroma |
| RAG Framework | LangChain |
| Backend API | FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyPDF / PyMuPDF |
| Configuration | Pydantic Settings |
| Testing | Pytest |

---

# Requirements

The main dependencies are:

```text
fastapi
uvicorn
streamlit
requests

langchain
langchain-core
langchain-chroma
langchain-huggingface
langchain-ollama

chromadb
sentence-transformers

pypdf
pymupdf

pydantic
pydantic-settings

python-dotenv

pytest
httpx
```

Install them with:

```powershell
pip install -r requirements.txt
```

---

# System Requirements

The application is designed to run locally.

Recommended components:

- Python 3.10+
- Ollama
- Sufficient RAM for the local embedding model and LLM
- Internet connection for initial model/package downloads

The application does not require a dedicated GPU for the current setup.

---

# Ollama Setup

Install Ollama on the operating system.

Verify the installation:

```powershell
ollama --version
```

Pull the required model:

```powershell
ollama pull qwen2.5:3b
```

Verify that the model is available:

```powershell
ollama list
```

The expected model is:

```text
qwen2.5:3b
```

Make sure the Ollama service is running before asking questions through the application.

---

# Hugging Face Setup

The project uses the Hugging Face model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded automatically when it is first required.

A Hugging Face token is optional for basic usage.

If you want to configure a token, create a `.env` file in the project root:

```text
HF_TOKEN=your_huggingface_token_here
```

Do not commit the `.env` file to Git.

The `.gitignore` file excludes:

```text
.env
```

from version control.

---

# Environment Configuration

Create:

```text
.env
```

in the project root.

Example:

```env
APP_NAME=AI/ML Research Navigator
APP_ENV=development
LOG_LEVEL=INFO

LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=800
CHUNK_OVERLAP=150

TOP_K=6
DISTANCE_THRESHOLD=1.15

COLLECTION_NAME=research_papers

HF_TOKEN=
```

If a Hugging Face token is available:

```env
HF_TOKEN=your_huggingface_token_here
```

Do not publish the token in GitHub or other public repositories.

---

# Data Directory

Place the research papers inside:

```text
data/papers/
```

For the current project:

```text
data/
└── papers/
    ├── resnet.pdf
    ├── vit.pdf
    └── swin_transformer.pdf
```

The path:

```text
research-rag/data/papers/
```

is the correct location when viewed from the project root.

---

# Document Ingestion

After placing the PDFs in the `data/papers/` directory, run the ingestion process:

```powershell
python -m app.ingestion
```

The ingestion process:

```text
PDF files
   ↓
PDF text extraction
   ↓
Document chunks
   ↓
Embedding generation
   ↓
Chroma vector database
```

After successful ingestion, the Chroma database is stored locally.

The generated database directory is:

```text
chroma_db/
```

This directory is intentionally excluded from Git because it is generated data.

---

# RAG Retrieval Process

When the user asks a question, the system performs the following process:

```text
Question
   ↓
Embedding
   ↓
Chroma similarity search
   ↓
Top-K candidate chunks
   ↓
Distance threshold filtering
   ↓
Relevant chunks
   ↓
Context construction
   ↓
LLM
   ↓
Grounded answer
```

The current configuration retrieves:

```text
TOP_K = 6
```

candidate chunks.

The relevance threshold is:

```text
DISTANCE_THRESHOLD = 1.15
```

The underlying retrieval can therefore examine multiple chunks while the application filters out chunks that do not satisfy the configured relevance condition.

---

# Context Construction

Each retrieved document chunk is converted into structured context containing:

```text
SOURCE
Paper
Page
Content
```

Conceptually:

```text
SOURCE 1
Paper: resnet.pdf
Page: 1

Content:
...
```

This gives the LLM information about where the retrieved content came from while keeping the answer generation grounded in the retrieved text.

---

# Answer Generation

The application uses Qwen 2.5 3B through Ollama.

The generation temperature is:

```text
temperature = 0
```

This is used to encourage consistent and deterministic responses.

The LLM receives:

```text
Research paper context
+
User question
```

and generates an answer based only on the supplied context.

---

# Out-of-Domain Handling

The application is designed not to answer questions that are unsupported by the research-paper knowledge base.

For example:

```text
What is the architecture of GPT-5?
```

The available papers do not provide information about GPT-5.

Therefore, the expected behavior is:

```text
The available research papers do not contain enough information to answer this question.
```

This is an important characteristic of the application because the system is intended to remain grounded in its local research-paper knowledge base rather than relying on the LLM's general knowledge.

---

# FastAPI Backend

The backend exposes the RAG functionality through FastAPI.

The development server can be started using:

```powershell
uvicorn app.main:app --reload
```

The API runs by default at:

```text
http://127.0.0.1:8000
```

The main question-answering endpoint is:

```text
POST /ask
```

The request contains the user's question.

Example:

```json
{
  "question": "What is the main idea behind residual learning?"
}
```

The API returns the generated answer and source information.

---

# Streamlit Frontend

The Streamlit frontend provides a user-friendly interface for interacting with the RAG system.

Start Streamlit using:

```powershell
streamlit run streamlit_app.py
```

The browser interface allows users to:

1. Enter a research question.
2. Submit the question.
3. Wait for retrieval and answer generation.
4. View the generated answer.
5. View the relevant source papers.
6. View deduplicated and numerically sorted page numbers.

Example:

```text
📄 resnet.pdf — Pages: 1, 2, 3
```

The frontend deduplicates source pages before displaying them.

Therefore, if the same paper page is retrieved multiple times, the user sees the page only once.

---

# Source Display

The retrieval process may internally retrieve multiple chunks from the same page.

For example:

```text
resnet.pdf — Page 2
resnet.pdf — Page 2
resnet.pdf — Page 3
resnet.pdf — Page 1
```

The Streamlit interface presents them as:

```text
📄 resnet.pdf — Pages: 1, 2, 3
```

This improves readability without changing the underlying retrieval process.

The frontend also sorts numeric page numbers correctly.

---

# Testing

The project uses Pytest for automated testing.

Run:

```powershell
pytest
```

The test suite is used to verify important project components such as:

- configuration
- database functionality
- API behavior
- RAG functionality
- investigation/retrieval components where applicable

All tests should be run after making significant changes to the application.

---

# Answer-Quality Evaluation

The project contains an answer-quality evaluation pipeline.

Run:

```powershell
python evaluation/evaluate_answers.py
```

The evaluation checks:

- answerable questions
- answer correctness
- keyword coverage
- unanswerable questions
- correct refusals

The current evaluation dataset contains:

```text
15 questions
```

including:

```text
11 answerable questions
4 unanswerable questions
```

---

# Current Answer-Quality Evaluation Results

The latest evaluation run produced:

```text
======================================================================
ANSWER-QUALITY RESULTS
======================================================================
Total questions: 15
Answerable questions: 11
Answerable questions passed: 11
Answerable accuracy: 100.00%
Average keyword coverage: 78.79%
Unanswerable questions: 4
Correct refusals: 4
Refusal accuracy: 100.00%
======================================================================
```

The latest evaluation also reported:

```text
======================================================================
FAILED ANSWERABLE QUESTIONS
======================================================================
No failed answerable questions.
======================================================================
```

These numbers describe the **current evaluation run** and should not be interpreted as a permanent guarantee.

If the evaluation dataset, retrieval configuration, prompt, embedding model, or LLM is changed, the evaluation should be run again and the results in this README should be updated.

---

# Retrieval Evaluation

A previous retrieval evaluation achieved:

```text
Total questions: 15
Correct retrievals: 15
Retrieval accuracy: 100.00%
```

This indicates that all 15 evaluation questions retrieved relevant document content according to the retrieval evaluation used at that stage.

The project currently contains the answer-quality evaluation script:

```text
evaluation/evaluate_answers.py
```

There is not currently a separate:

```text
evaluate_retrieval.py
```

file.

Therefore, retrieval evaluation should not be represented in the project structure as a separate script unless one is added later.

---

# Evaluation Interpretation

The current evaluation results indicate:

### Answerable accuracy

```text
100.00%
```

All 11 answerable questions in the current evaluation dataset passed the evaluation criteria.

### Keyword coverage

```text
78.79%
```

Keyword coverage measures how much of the expected evaluation vocabulary was reflected in the generated answers.

A score of 78.79% does not mean that 78.79% of the answer is correct. It is a separate evaluation metric.

### Refusal accuracy

```text
100.00%
```

All four unanswerable questions received the expected refusal behavior.

This is particularly important for a grounded RAG system because it demonstrates that the current pipeline can reject questions that are outside the available research-paper context.

### Retrieval accuracy

The previous retrieval evaluation achieved:

```text
100.00%
```

This indicates that the tested questions successfully retrieved relevant document content according to that evaluation.

---

# Example Questions

The following questions can be used to test the system.

## ResNet

```text
What is the main idea behind residual learning?
```

```text
What does the residual function represent in ResNet?
```

```text
How are shortcut connections used in ResNet?
```

```text
What problem does ResNet attempt to address?
```

## Vision Transformer

```text
How does Vision Transformer process an image?
```

```text
What is the role of patch embeddings in ViT?
```

```text
How are image patches represented in ViT?
```

```text
Why does Vision Transformer divide an image into patches?
```

```text
How does ViT differ from traditional convolutional neural networks?
```

## Swin Transformer

```text
What is the purpose of shifted windows in Swin Transformer?
```

## Out-of-Domain

```text
What is the architecture of GPT-5?
```

The expected behavior for unsupported questions is:

```text
The available research papers do not contain enough information to answer this question.
```

---

# Running the Complete Application

The recommended workflow is:

## Step 1 — Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If the virtual environment does not exist:

```powershell
python -m venv .venv
```

Then activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 3 — Verify Ollama

```powershell
ollama --version
```

Verify the Qwen model:

```powershell
ollama list
```

If necessary:

```powershell
ollama pull qwen2.5:3b
```

---

## Step 4 — Add research papers

Place the three PDFs inside:

```text
data/papers/
```

Example:

```text
data/papers/resnet.pdf
data/papers/vit.pdf
data/papers/swin_transformer.pdf
```

---

## Step 5 — Run ingestion

```powershell
python -m app.ingestion
```

Wait for the documents to be processed and the Chroma database to be created.

---

## Step 6 — Start FastAPI

Open a terminal:

```powershell
uvicorn app.main:app --reload
```

Keep this terminal running.

---

## Step 7 — Start Streamlit

Open another terminal and activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```powershell
streamlit run streamlit_app.py
```

Open the Streamlit URL displayed in the terminal.

---

## Step 8 — Test the application

Ask:

```text
What is the main idea behind residual learning?
```

Then test:

```text
How does Vision Transformer process an image?
```

Then:

```text
What is the purpose of shifted windows in Swin Transformer?
```

Finally test the out-of-domain behavior:

```text
What is the architecture of GPT-5?
```

---

# Configuration

The main configuration is stored in:

```text
app/config.py
```

Important settings include:

```text
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 6

DISTANCE_THRESHOLD = 1.15
```

The application also loads environment-specific values from:

```text
.env
```

The configuration uses Pydantic Settings.

---

# Security and Git Configuration

The project `.gitignore` excludes sensitive and generated files such as:

```text
.env
chroma_db/
*.db
*.sqlite
*.sqlite3
__pycache__/
.pytest_cache/
.venv/
```

The Hugging Face token must never be hard-coded into Python source files.

Use:

```text
.env
```

instead.

---

# .gitignore

The current `.gitignore` includes:

```gitignore
# Python cache
__pycache__/

# Virtual environment
.venv/

# Environment variables / secrets
.env

# Chroma vector database
chroma_db/
```

---

# Design Goals

The project is designed around the following principles:

### 1. Grounded Generation

Answers should be based on retrieved research-paper content rather than unsupported LLM knowledge.

### 2. Local LLM Inference

Qwen 2.5 3B is executed locally using Ollama.

### 3. Semantic Retrieval

Research-paper chunks are retrieved using vector similarity rather than simple keyword matching.

### 4. Relevance Filtering

Retrieved chunks are filtered using a configurable distance threshold.

### 5. Source Transparency

The application displays the research papers and pages associated with the retrieved context.

### 6. Out-of-Domain Refusal

The system should refuse to answer when the available research-paper context does not contain sufficient information.

### 7. Evaluation

The project includes an answer-quality evaluation pipeline to measure answerable-question performance and refusal behavior.

---

# Possible Future Improvements

Potential future improvements include:

- Hybrid keyword + vector retrieval
- Reranking retrieved chunks
- Query expansion
- Better citation mapping
- Page-level PDF viewing
- Conversation history
- Multi-document comparison
- Additional research papers
- More extensive evaluation datasets
- Retrieval-specific evaluation tooling
- Automated document management
- Improved chunking strategies
- Metadata filtering
- Streaming LLM responses
- Docker deployment
- Cloud deployment

These are potential future improvements and are not currently implemented unless explicitly added to the project.

---

# What This Project Demonstrates

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation
- Large Language Models
- Local LLM inference
- Embeddings
- Vector databases
- Semantic search
- Document ingestion
- PDF processing
- Prompt engineering
- Grounded generation
- Hallucination control
- Relevance filtering
- FastAPI
- Streamlit
- LangChain
- Chroma
- Hugging Face
- Ollama
- Evaluation methodology
- Environment configuration

---

# Conclusion

AI/ML Research Navigator is an end-to-end local RAG application designed to answer questions about computer vision and machine learning research papers.

The project combines:

```text
PDF Documents
      ↓
Document Processing
      ↓
Chunking
      ↓
Hugging Face Embeddings
      ↓
Chroma Vector Database
      ↓
Semantic Retrieval
      ↓
Relevance Filtering
      ↓
Context Construction
      ↓
Qwen 2.5 3B
      ↓
Grounded Answer
      ↓
FastAPI
      ↓
Streamlit
```

The current system successfully supports research-paper question answering across the ResNet, Vision Transformer, and Swin Transformer papers.

The latest answer-quality evaluation achieved:

```text
Answerable accuracy: 100.00%
Average keyword coverage: 78.79%
Refusal accuracy: 100.00%
```

The previous retrieval evaluation achieved:

```text
Retrieval accuracy: 100.00%
```

These results represent the evaluation datasets and pipeline configuration used for the current project and should be re-evaluated after significant changes.

---

# Author

**Hamza totss**

AI/ML | Data Science | AI BigData Engineer
