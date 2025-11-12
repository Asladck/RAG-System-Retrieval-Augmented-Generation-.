import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import docx
import requests
from dotenv import load_dotenv

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"
load_dotenv()

def generate_with_ollama(prompt: str, model: str = "llama3", max_tokens: int = 512, temperature: float = 0.0, host: Optional[str] = None) -> str:
    import json

    host = host or os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        with requests.post(f"{host}/api/generate", json=payload, stream=True, timeout=120) as resp:
            if not resp.ok:
                return f"(Ollama returned status {resp.status_code}): {resp.text}"

            output = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output.append(data["response"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return "".join(output).strip() or "(No output from Ollama)"
    except Exception as e:
        return f"(Ollama error: {e}). Убедитесь, что Ollama запущен и доступен по {host}."

class Ingestor:
    """Loads documents from a directory, extracts text, chunks, and builds/saves a FAISS index or a numpy-backed index."""

    def __init__(self, source_dir: str, index_path: str = "./index.faiss", embed_model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 80, overlap: int = 15):
        self.source_dir = Path(source_dir)
        self.index_path = Path(index_path)
        self.embed_model_name = embed_model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = SentenceTransformer(self.embed_model_name)
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    def extract_text_from_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf(path)
        if path.suffix.lower() == ".docx":
            return self._extract_docx(path)
        if path.suffix.lower() in [".txt", ".md"]:
            return path.read_text(encoding="utf-8")
        return ""

    def _extract_pdf(self, path: Path) -> str:
        texts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        return "\n".join(texts)

    def _extract_docx(self, path: Path) -> str:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    def preprocess(self, text: str) -> str:
        return " ".join(text.split())

    def chunk_text(self, text: str) -> List[str]:
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []

        for paragraph in paragraphs:
            words = paragraph.split()

            if len(words) <= self.chunk_size:
                if words:
                    chunks.append(paragraph)
                continue
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                if chunk_words:
                    chunk_text = ' '.join(chunk_words)
                    chunks.append(chunk_text)

        return chunks

    def ingest(self):
        files = list(self.source_dir.glob("**/*.*"))
        for f in files:
            text = self.extract_text_from_file(f)
            if not text:
                continue
            text = self.preprocess(text)
            chunks = self.chunk_text(text)
            for idx, c in enumerate(chunks):
                self.texts.append(c)
                self.metadatas.append({"source": str(f), "chunk": idx})
        if not self.texts:
            print("No documents found to ingest.")
            return

        embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)

        # Save texts and metadata
        with open(str(self.index_path) + ".texts.txt", "w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(t.replace("\n", " ") + "\n")
        import json
        with open(str(self.index_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

        if _FAISS_AVAILABLE:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            faiss.write_index(index, str(self.index_path))
            print(f"Ingested {len(self.texts)} chunks. FAISS index saved to {self.index_path}")
        else:
            # fallback: save embeddings to .npy and note that FAISS is unavailable
            np.save(str(self.index_path) + ".embeddings.npy", embeddings)
            print(f"Ingested {len(self.texts)} chunks. FAISS not available — embeddings saved to {self.index_path}.embeddings.npy")


class RAGSystem:
    def __init__(self, index_path: str = "./index.faiss", embed_model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = Path(index_path)
        self.model = SentenceTransformer(embed_model_name)
        self.use_faiss = False
        import json
        meta_file = str(self.index_path) + ".meta.json"
        texts_file = str(self.index_path) + ".texts.txt"
        emb_file = str(self.index_path) + ".embeddings.npy"

        if not (Path(meta_file).exists() and Path(texts_file).exists()):
            raise FileNotFoundError("Index metadata or texts file not found; run ingest first")

        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        with open(texts_file, "r", encoding="utf-8") as f:
            self.texts = [line.rstrip('\n') for line in f.readlines()]

        if _FAISS_AVAILABLE and Path(str(self.index_path)).exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.use_faiss = True
            except Exception:
                self.use_faiss = False

        if not self.use_faiss:
            if Path(emb_file).exists():
                self.embeddings = np.load(emb_file)
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embeddings = self.embeddings / norms
            else:
                raise FileNotFoundError("No FAISS index and no embeddings found; run ingest first")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[float, dict, str]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        if not self.use_faiss:
            q_norm = q_emb[0] / (np.linalg.norm(q_emb[0]) + 1e-12)
            sims = (self.embeddings @ q_norm)
            idxs = np.argsort(-sims)[:k]
            results = []
            for idx in idxs:
                score = float(sims[idx])
                md = self.metadatas[int(idx)]
                txt = self.texts[int(idx)]
                results.append((score, md, txt))
            return results
        else:
            D, I = self.index.search(q_emb, k)
            results = []
            for score, idx in zip(D[0], I[0]):
                md = self.metadatas[int(idx)]
                txt = self.texts[int(idx)]
                results.append((float(score), md, txt))
            return results

    def answer(self, query: str, k: int = 3) -> str:
        results = self.retrieve(query, k)
        if not results:
            return "Answer:\n(No relevant documents found)\n\nSources:\n- none"
        context = "\n\n".join([r[2] for r in results])
        if all((txt.startswith('See ') or txt.strip() == '') for (_, _, txt) in results):
            return "Answer:\n(No relevant documents found)\n\nSources:\n" + "\n".join([f"- {r[1].get('source')} (chunk {r[1].get('chunk')})" for r in results])
        sources = [r[1] for r in results]

        use_ollama = os.environ.get("USE_OLLAMA") == "1" or os.environ.get("USE_OLLAMA") == "true"
        if not use_ollama and os.environ.get("OLLAMA_HOST"):
            use_ollama = True

        ans = None
        if use_ollama:
            prompt = f"Answer in one short sentence based on context; if not found, say 'I don't know'. Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ans = generate_with_ollama(prompt, model=os.environ.get("OLLAMA_MODEL", "llama3"), max_tokens=512, temperature=0.0, host=os.environ.get("OLLAMA_HOST"))

        if ans and not ans.startswith("(Ollama") and ans.strip():
            src_text = "\n".join([f"- {s.get('source')} (chunk {s.get('chunk')})" for s in sources])
            return f"Answer:\n{ans}\n\nSources:\n{src_text} \n Original context used:\n{context}"

        ans = "(Answer based on retrieved documents but no LLM processing available)"
        src_text = "\n".join([f"- {s.get('source')} (chunk {s.get('chunk')})" for s in sources])
        return f"Answer:\n{ans}\n\nSources:\n{src_text} \n Original context used:\n{context}"
