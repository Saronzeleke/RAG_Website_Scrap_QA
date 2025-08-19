import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder  
from setting import settings
import logging
import numpy as np
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self):
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection("visitethiopia_docs")
        except Exception:
            self.collection = self.client.create_collection("visitethiopia_docs")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if settings.use_reranking else None  # Init cross-encoder
        self.bm25 = None
        self.documents = []

    def clear(self):
        try:
            self.client.delete_collection("visitethiopia_docs")
        except Exception:
            pass
        self.__init__()

    def add_documents(self, documents: list):
        try:
            if not documents:
                return
            filtered_docs = [doc for doc in documents if doc.get("content")]
            if not filtered_docs:
                return
            texts = [doc["content"] for doc in filtered_docs]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.bm25 = BM25Okapi([t.split() for t in texts])
            
            start_idx = len(self.documents)
            ids = []
            docs_list = []  
            embs = []
            metas = []
            
            for i, doc in enumerate(filtered_docs):
                idx = start_idx + i
                chroma_id = f"idx_{idx}"
                ids.append(chroma_id)
                docs_list.append(doc["content"])
                embs.append(embeddings[i].tolist())
                meta = {
                    "doc_idx": idx,
                    "source": doc.get("source"),
                    "title": doc.get("title", ""),
                    "updated_at": doc.get("updated_at", "")
                }
                metas.append(meta)
                self.documents.append(doc)
            
            self.collection.add(
                ids=ids,
                documents=docs_list,
                embeddings=embs,
                metadatas=metas
            )
            logger.info(f"Added {len(ids)} docs to Chroma (total docs={len(self.documents)})")
        except Exception as e:
            logger.error(f"Error adding documents to retrieval: {e}")

def _chunk_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > settings.chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_documents(documents: list) -> list:
    chunks = []
    for orig_idx, doc in enumerate(documents):
        content = doc.get("content", {})
        if isinstance(content, dict):
            text_chunks = _chunk_text(content.get("main_text", ""))
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "orig_doc_idx": orig_idx,
                    "chunk_index": i,
                    "id": f"{orig_idx}_text_{i}",
                    "title": doc.get("title", ""),
                    "content": chunk,
                    "source": doc.get("source", ""),
                    "metadata": {
                        "type": "main_text",
                        "headings": content.get("metadata", {}).get("headings", [])[:3]
                    }
                })
            for table in content.get("metadata", {}).get("tables", []):
                table_chunk = {
                    "orig_doc_idx": orig_idx,
                    "chunk_index": len(chunks),
                    "id": f"{orig_idx}_table_{len(chunks)}",
                    "title": f"Table: {table.get('caption', '')}",
                    "content": "\n".join(["|".join(row) for row in table.get("rows", [])]),
                    "source": doc.get("source", ""),
                    "metadata": {
                        "type": "table",
                        "caption": table.get("caption", "")
                    }
                }
                chunks.append(table_chunk)
        else:
            text_chunks = _chunk_text(str(content))
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "orig_doc_idx": orig_idx,
                    "chunk_index": i,
                    "id": f"{orig_idx}_text_{i}",
                    "title": doc.get("title", ""),
                    "content": chunk,
                    "source": doc.get("source", ""),
                    "metadata": {"type": "plain_text"}
                })
    return chunks

async def _expand_query(query: str) -> List[str]:
    expansions = [query]
    if "hotel" in query.lower():
        expansions.extend([query + " accommodation", query + " lodging"])
    if "tour" in query.lower():
        expansions.extend([query + " package", query + " travel"])
    seen = set()
    return [x for x in expansions if not (x in seen or seen.add(x))]

async def _rerank_with_cross_encoder(query: str, results: List[Dict], cross_encoder) -> List[Dict]:
    if not cross_encoder or len(results) < 2:
        return results
    try:
        pairs = [(query, doc['content'][:512]) for doc in results]
        scores = cross_encoder.predict(pairs)
        for doc, score in zip(results, scores):
            doc['score'] = float(score)
        return sorted(results, key=lambda x: x['score'], reverse=True)
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return results

async def hybrid_retrieval(query: str, retrieval: RetrievalSystem, k: int = 5) -> list:
    try:
        if not retrieval or not retrieval.documents:
            return []
        expanded_queries = await _expand_query(query)
        bm25_scores = []
        for q in expanded_queries:
            scores = retrieval.bm25.get_scores(q.split()) if retrieval.bm25 else np.zeros(len(retrieval.documents))
            bm25_scores.append(scores)
        bm25_combined = np.max(bm25_scores, axis=0)
        
        query_embeddings = retrieval.model.encode(expanded_queries)
        chroma_results = retrieval.collection.query(
            query_embeddings=[e.tolist() for e in query_embeddings],
            n_results=k*2
        )
        
        results = []
        seen_idxs = set()
        for ids_row, dists_row, metas_row in zip(chroma_results["ids"], chroma_results["distances"], chroma_results["metadatas"]):
            for id_str, dist, meta in zip(ids_row, dists_row, metas_row):
                doc_idx = meta.get("doc_idx")
                if doc_idx is None or doc_idx in seen_idxs:
                    continue
                similarity = 1.0 - float(dist)
                bm25_score = float(bm25_combined[doc_idx]) if doc_idx < len(bm25_combined) else 0.0
                denom = float(np.max(bm25_combined)) if np.any(bm25_combined) else 1.0
                positional_boost = 1.0 - (doc_idx / len(retrieval.documents)) * 0.2
                combined_score = (0.5 * similarity + 0.3 * (bm25_score / denom) + 0.2 * positional_boost)
                doc = dict(retrieval.documents[doc_idx])
                doc["score"] = combined_score
                results.append(doc)
                seen_idxs.add(doc_idx)
        
        if settings.use_reranking and len(results) > 1:
            results = await _rerank_with_cross_encoder(query, results, retrieval.cross_encoder)
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    except Exception as e:
        logger.error(f"Error in hybrid_retrieval: {e}")
        return []