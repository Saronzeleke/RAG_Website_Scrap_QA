
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from setting import settings
import logging
import numpy as np
import numpy as np
from typing import List, Dict
import logging
from sentence_transformers import cross_encoder
import re
logger = logging.getLogger(__name__)

class RetrievalSystem:
    """
    Persistent retrieval system using Chroma + SentenceTransformer + BM25
    Keep one instance in app.state.retrieval
    """
    def __init__(self):
        self.client = chromadb.Client()
        # get or create deterministic collection
        try:
            self.collection = self.client.get_collection("visitethiopia_docs")
        except Exception:
            self.collection = self.client.create_collection("visitethiopia_docs")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.bm25 = None
        # list of documents (original doc/chunk objects) used for mapping
        self.documents = []

    def clear(self):
        try:
            self.client.delete_collection("visitethiopia_docs")
        except Exception:
            pass
        # re-init
        self.__init__()

    def add_documents(self, documents: list):
        """
        documents: list of dicts with fields: id, title, content, source, updated_at
        We'll assign stable chroma ids "idx_{i}" and store doc_idx in metadata.
        """
        try:
            if not documents:
                return
            texts = [doc.get("content", "") for doc in documents if doc.get("content")]
            if not texts:
                return

            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # build BM25 over all texts (order must match documents passed)
            self.bm25 = BM25Okapi([t.split() for t in texts])

            start_idx = len(self.documents)
            ids = []
            docs = []
            embs = []
            metas = []

            for i, doc in enumerate(documents):
                idx = start_idx + i
                chroma_id = f"idx_{idx}"
                ids.append(chroma_id)
                docs.append(doc.get("content", ""))
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
                documents=docs,
                embeddings=embs,
                metadatas=metas
            )
            logger.info(f"Added {len(ids)} docs to Chroma (total docs={len(self.documents)})")
        except Exception as e:
            logger.error(f"Error adding documents to retrieval: {e}")
def _chunk_text(text: str) -> List[str]:
    """Split text into chunks respecting sentence boundaries"""
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
    """Content-aware chunking that preserves structure"""
    chunks = []
    for orig_idx, doc in enumerate(documents):
        content = doc.get("content", {})
        if isinstance(content, dict):  # Handle our enhanced content format
            # Chunk main text
            text_chunks = _chunk_text(content.get("main_text", ""))
            
            # Create chunks with metadata
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
            
            # Create special chunks for tables
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
        else:  # Fallback for plain text content
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
# def chunk_documents(documents: list) -> list:
#     """
#     Deterministic chunking that keeps orig_doc_idx for mapping.
#     Each returned chunk includes:
#      - orig_doc_idx (index of source document in input list)
#      - chunk_index
#      - id (string)
#      - content, title, source, updated_at
#     """
#     chunks = []
#     for orig_idx, doc in enumerate(documents):
#         text = doc.get("content", "")
#         if not text:
#             continue
#         words = text.split()
#         step = settings.chunk_size - settings.chunk_overlap
#         if step <= 0:
#             step = settings.chunk_size
#         i = 0
#         chunk_i = 0
#         while i < len(words):
#             chunk_text = " ".join(words[i:i + settings.chunk_size])
#             chunks.append({
#                 "orig_doc_idx": orig_idx,
#                 "chunk_index": chunk_i,
#                 "id": f"{orig_idx}_{chunk_i}",
#                 "title": doc.get("title", ""),
#                 "content": chunk_text,
#                 "source": doc.get("source", ""),
#                 "updated_at": doc.get("updated_at", "")
#             })
#             i += step
#             chunk_i += 1
#     return chunks
async def _expand_query(query: str) -> List[str]:
    """Generate query variations to improve retrieval"""
    expansions = [query]  # Always include original query
    
    # Simple rule-based expansions
    if "hotel" in query.lower():
        expansions.extend([query + " accommodation", query + " lodging"])
    if "tour" in query.lower():
        expansions.extend([query + " package", query + " travel"])
    
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in expansions if not (x in seen or seen.add(x))]
async def _rerank_with_cross_encoder(query: str, results: List[Dict]) -> List[Dict]:
    """Re-rank results using cross-encoder model"""
    if not cross_encoder or len(results) < 2:
        return results
    
    try:
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, doc['content'][:512]) for doc in results]  # Truncate content
        
        # Get scores from cross-encoder
        scores = cross_encoder.predict(pairs)
        
        # Update document scores
        for doc, score in zip(results, scores):
            doc['score'] = float(score)  # Override previous score
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return results
async def hybrid_retrieval(query: str, retrieval: RetrievalSystem, k: int = 5) -> list:
    """Enhanced hybrid retrieval with query expansion"""
    try:
        if not retrieval or not retrieval.documents:
            return []
            
        # Query expansion
        expanded_queries = await _expand_query(query)
        
        # Get BM25 scores for each variant
        bm25_scores = []
        for q in expanded_queries:
            scores = retrieval.bm25.get_scores(q.split()) if retrieval.bm25 else np.zeros(len(retrieval.documents))
            bm25_scores.append(scores)
        bm25_combined = np.max(bm25_scores, axis=0)
        
        # Get embeddings for each variant
        query_embeddings = retrieval.model.encode(expanded_queries)
        
        # Query Chroma with multiple embeddings
        chroma_results = retrieval.collection.query(
            query_embeddings=[e.tolist() for e in query_embeddings],
            n_results=k*2  # Get more results to re-rank
        )
        
        results = []
        seen_idxs = set()

        for ids_row, dists_row, metas_row in zip(chroma_results["ids"], chroma_results["distances"], chroma_results["metadatas"]):
            for id_str, dist, meta in zip(ids_row, dists_row, metas_row):
                doc_idx = meta.get("doc_idx")
                if doc_idx is None or doc_idx in seen_idxs:
                    continue
                    
                # Calculate combined score
                similarity = 1.0 - float(dist)
                bm25_score = float(bm25_combined[doc_idx]) if doc_idx < len(bm25_combined) else 0.0
                denom = float(np.max(bm25_combined)) if np.any(bm25_combined) else 1.0
                
                # Positional boost for earlier documents (assuming more important content comes first)
                positional_boost = 1.0 - (doc_idx / len(retrieval.documents)) * 0.2
                
                combined_score = (0.5 * similarity + 
                                0.3 * (bm25_score / denom) + 
                                0.2 * positional_boost)
                
                doc = dict(retrieval.documents[doc_idx])
                doc["score"] = combined_score
                results.append(doc)
                seen_idxs.add(doc_idx)
                
        # Re-rank with cross-encoder if enabled
        if settings.use_reranking and len(results) > 1:
            results = await _rerank_with_cross_encoder(query, results)
            
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    except Exception as e:
        logger.error(f"Error in hybrid_retrieval: {e}")
        return []
# async def hybrid_retrieval(query: str, retrieval: RetrievalSystem, k: int = 5) -> list:
#     """
#     Query the persisted RetrievalSystem for top-k relevant documents.
#     Returns list of document dicts with 'score' field added.
#     """
#     try:
#         if not retrieval or not retrieval.documents:
#             return []
#         # bm25 scores (length = number of docs)
#         bm25_scores = retrieval.bm25.get_scores(query.split()) if retrieval.bm25 is not None else np.zeros(len(retrieval.documents))

#         query_embedding = retrieval.model.encode([query])[0]
#         chroma_results = retrieval.collection.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=k
#         )
#         results = []
#         seen_idxs = set()

#         # chroma_results keys: 'ids', 'distances', 'metadatas'
#         for ids_row, dists_row, metas_row in zip(chroma_results["ids"], chroma_results["distances"], chroma_results["metadatas"]):
#             for id_str, dist, meta in zip(ids_row, dists_row, metas_row):
#                 doc_idx = meta.get("doc_idx")
#                 if doc_idx is None:
#                     continue
#                 if doc_idx in seen_idxs:
#                     continue
#                 # similarity approximation
#                 similarity = 1.0 - float(dist)
#                 bm25_score = float(bm25_scores[doc_idx]) if doc_idx < len(bm25_scores) else 0.0
#                 denom = float(np.max(bm25_scores)) if np.any(bm25_scores) else 1.0
#                 combined_score = 0.6 * similarity + 0.4 * (bm25_score / denom)
#                 doc = dict(retrieval.documents[doc_idx])
#                 doc["score"] = combined_score
#                 results.append(doc)
#                 seen_idxs.add(doc_idx)
#         results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
#         return results_sorted
#     except Exception as e:
#         logger.error(f"Error in hybrid_retrieval: {e}")
#         return []

