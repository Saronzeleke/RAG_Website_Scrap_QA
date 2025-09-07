from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict
import logging
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import nltk
import hashlib
nltk.download('punkt', quiet=True)

class Retriever:
    def __init__(self, embedding_model: str, cross_encoder_model: str, 
                 chunk_size: int, chunk_overlap: int):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.client = Client(Settings(persist_directory="./chroma_db", anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="visitethiopia",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )
        
        # BM25 on chunks
        self.bm25_corpus = [] 
        self.bm25 = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Retriever initialized with {embedding_model} and {cross_encoder_model}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using sentence boundaries"""
        if not text or not text.strip():
            return []
            
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = len(current_chunk)
            
            current_chunk.extend(words)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def add_documents(self, documents: List[Dict]):
        """Add or update documents in both vector store and BM25 index"""
        if not documents:
            return
            
        # First, remove existing versions
        doc_ids = [doc['id'] for doc in documents]
        self.remove_documents(doc_ids)
            
        texts = []
        metadatas = []
        ids = []
        new_bm25_chunks = []
        
        for doc in documents:
            content = doc.get('content', '') or ''
            if not content.strip():
                continue
                
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                chunk_id = f"{doc['id']}_{i}_{chunk_hash}"
                texts.append(chunk)
                metadatas.append({
                    'id': doc['id'],
                    'table_name': doc.get('table_name', ''),
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'chunk_index': i,
                    'original_doc_id': doc['id']
                })
                ids.append(chunk_id)
                new_bm25_chunks.append(chunk)
        
        if texts:
            # Add to vector store
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update BM25
            self.bm25_corpus.extend(new_bm25_chunks)
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            self.logger.info(f"Added/Updated {len(texts)} chunks from {len(documents)} documents")

    def remove_documents(self, doc_ids: List[str]):
        """Remove documents by their original IDs"""
        if not doc_ids:
            return
            
        records = self.collection.get()
        ids_to_remove = [
            id_ for id_, metadata in zip(records['ids'], records['metadatas'])
            if metadata.get('original_doc_id') in doc_ids
        ]
        
        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
            
            # Rebuild BM25 from remaining
            remaining_records = self.collection.get()
            self.bm25_corpus = remaining_records['documents']
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            
            self.logger.info(f"Removed {len(ids_to_remove)} chunks for {len(doc_ids)} documents")

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.18, 
                semantic_weight: float = 0.5, bm25_weight: float = 0.3, 
                positional_weight: float = 0.2) -> List[Dict]:
        """Hybrid retrieval with semantic + BM25 + re-ranking"""
        
        # 1. Semantic Search
        query_embedding = self.embedding_model.encode(query).tolist()
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3  
        )
        
        semantic_docs = semantic_results['metadatas'][0]
        semantic_scores = [1 - dist for dist in semantic_results['distances'][0]]  
        semantic_ids = semantic_results['ids'][0]
        
        # 2. BM25 Search (if corpus exists)
        bm25_scores = []
        if self.bm25 and self.bm25_corpus:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Map BM25 scores to the same documents as semantic search
            bm25_mapped_scores = []
            for id_ in semantic_ids:
                try:
                    idx = self.bm25_corpus.index(self.collection.get(ids=[id_])['documents'][0])
                    bm25_mapped_scores.append(bm25_scores[idx])
                except (ValueError, IndexError):
                    bm25_mapped_scores.append(0)
        else:
            bm25_mapped_scores = [0] * len(semantic_ids)
        
        # 3. Combine scores
        combined_results = []
        for i, (doc, semantic_score, bm25_score, id_) in enumerate(
            zip(semantic_docs, semantic_scores, bm25_mapped_scores, semantic_ids)):
            
            positional_score = 1 / (i + 1)  
            final_score = (semantic_weight * semantic_score + 
                         bm25_weight * (bm25_score / max(bm25_mapped_scores + [1]) if bm25_mapped_scores else 0) +
                         positional_weight * positional_score)
            
            if final_score >= min_score:
                combined_results.append({
                    'metadata': doc,
                    'score': final_score,
                    'id': id_,
                    'content': self.collection.get(ids=[id_])['documents'][0]
                })
        
        # 4. Re-rank with cross-encoder
        if combined_results:
            pairs = [(query, f"{doc['metadata']['title']} {doc['content']}") 
                    for doc in combined_results]
            
            rerank_scores = self.cross_encoder.predict(pairs)
            
            for doc, rerank_score in zip(combined_results, rerank_scores):
                doc['rerank_score'] = rerank_score
                doc['final_score'] = doc['score'] * 0.3 + rerank_score * 0.7  
            
            combined_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return combined_results[:top_k]

    def get_document_count(self) -> int:
        """Get total number of chunks in the retriever"""
        return len(self.bm25_corpus) if self.bm25_corpus else 0