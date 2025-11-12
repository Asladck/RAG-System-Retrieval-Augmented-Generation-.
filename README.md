# 🧠 RAG System — Retrieval-Augmented Generation with Ollama + FAISS

> A local RAG (Retrieval-Augmented Generation) system that combines **semantic search** using **FAISS** and **answer generation** via **Ollama (Llama3)** or other LLMs.  
> Built with pure Python for educational purposes — simple, fast, and fully offline.

---

## 🚀 Features

✅ Loads and processes multiple document formats: **PDF, DOCX, TXT, MD**  
✅ Splits documents into **semantic chunks** with overlap  
✅ Generates **embeddings** using `SentenceTransformer` (`all-MiniLM-L6-v2`)  
✅ Stores vectors in **FAISS** for fast similarity search  
✅ Integrates with **Ollama (Llama3)** for local text generation  
✅ Provides a **CLI interface** for ingestion and querying  
✅ Displays **sources** of information in responses  
✅ Works fully **offline** — no paid APIs required  

---

## 🧩 System Architecture

```text
User Question
      │
      ▼
Text Embedding (SentenceTransformer)
      │
      ▼
Semantic Retrieval (FAISS / cosine similarity)
      │
      ▼
Prompt Construction (context + question)
      │
      ▼
Answer Generation (Ollama Llama3 / OpenAI / T5 fallback)
      │
      ▼
Final Answer + Source Documents
```

🛠️ Installation
1. Clone the repository
```bash
git clone https://github.com/<your-username>/rag-system.git
cd rag-system
```
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Install and run Ollama
Download Ollama from 👉 https://ollama.com/download

# Then run:
```bash
ollama serve
ollama pull llama3
```
## 📘 Usage
🏗️ 1. Ingest your documents
Place your documents into a folder (e.g., data/), then run:
```bash
python rag_cli.py ingest --source_dir data --index_path index_sdu
```
This will:

Read all PDF/DOCX/TXT/MD files
Split them into semantic chunks
Create embeddings using MiniLM
Store them in a FAISS index (index_sdu

🔍 2. Query the system
Ask questions about your knowledge base:
```bash
python rag_cli.py query --index_path index_sdu --k 3 "What are the main faculties at SDU?"
```
Example Output:
Answer:
```text
1. Faculty of Engineering and Natural Sciences
2. Faculty of Business and Law
3. Faculty of Education and Humanities
4. Faculty of Social Sciences
5. School of Liberal Arts and Design

Sources:
- data/sdu_course.txt
- data/sdu_info.pdf
- data/README.txt
```
⚙️ Project Structure
```graphql
📂 ASS3/
 ├── data/                     # Your input documents
 ├── rag.py                    # Core RAG logic (Ingestor + RAGSystem)
 ├── rag_cli.py                # CLI entrypoint
 ├── index_sdu*                # FAISS index files and metadata
 ├── requirements.txt          # Python dependencies
 └── README.md                 # This file 😎
```

🧠 Theoretical Overview
| Stage             | Description                                                                  |
| ----------------- | ---------------------------------------------------------------------------- |
| **Ingestion**     | Loads documents (PDF/DOCX/TXT), cleans and splits text, generates embeddings |
| **Vectorization** | Converts chunks into numerical vectors using SentenceTransformer             |
| **Storage**       | Stores vectors and metadata in FAISS for fast similarity search              |
| **Retrieval**     | Finds top-k most relevant chunks based on semantic similarity                |
| **Generation**    | Uses Ollama (Llama3) to generate a natural language answer                   |
| **Attribution**   | Displays source documents used for the answer                                |

🧪 Example Commands
```bash
# Ingest documents into an index
python rag_cli.py ingest --source_dir data --index_path index_sdu

# Ask a question with top-3 document retrieval
python rag_cli.py query --index_path index_sdu --k 3 "When was SDU founded?"

# Test ingestion and retrieval in one go
python smoke_test.py
```

🧰 Technologies Used

| Component               | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| **Python**              | Core programming language                        |
| **SentenceTransformer** | Creates embeddings for text                      |
| **FAISS**               | Vector database for fast semantic search         |
| **Ollama (Llama3)**     | Local large language model for answer generation |
| **pdfplumber**          | Extracts text from PDFs                          |
| **docx**                | Parses Word documents                            |
| **numpy**               | Math operations for embeddings                   |
| **argparse**            | Command-line interface                           |

🧾 License

This project is open-source and free to use under the MIT License.

👨‍💻 Author

Aibar Tlekbay
Retrieval-Augmented Generation System Developer
📍 Suleyman Demirel University
💬 “Local AI is the future — no cloud required.”
