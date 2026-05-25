import os
import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer
from app.config import settings

class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom embedding function wrapper for ChromaDB using local SentenceTransformers.
    Processes vector representations completely locally and offline.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Encode documents using the sentence transformer model
        embeddings = self.model.encode(list(input), show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

class VectorStoreManager:
    """
    Manages connections, collections, and query interactions with our local persistent ChromaDB.
    """

    def __init__(self):
        # Initialize persistent directory
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
        
        # Create persistent Chroma client
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        
        # Load local embedding function
        self.embedding_fn = SentenceTransformerEmbeddingFunction(settings.LOCAL_EMBEDDING_MODEL)
        
        # Set up core collection
        self.collection_name = "placement_intelligence"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for text matching
        )

    def reset_db(self):
        """Wipes the existing collections and starts fresh."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass # Collection did not exist yet
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list):
        """
        Pushes a list of structured chunks (dicts with 'text' and 'metadata') to ChromaDB.
        Performs batched insertion.
        """
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            ids.append(f"chunk_{idx}_{chunk['metadata'].get('section', 'gen')}")
            documents.append(chunk["text"])
            
            # Prepare metadata (Chroma DB only accepts string, int, float, or bool)
            meta = {}
            for key, val in chunk["metadata"].items():
                if isinstance(val, list):
                    # ChromaDB metadata cannot store list elements directly
                    # Serialize list or extract string
                    meta[key] = ",".join(str(v) for v in val)
                elif isinstance(val, (str, int, float, bool)):
                    meta[key] = val
            metadatas.append(meta)

        # Batch write in groups of 100 to optimize SQLite disk writes
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def query(self, query_text: str, n_results: int = 5, section_filter: str = None, company_filter: str = None) -> list:
        """
        Queries the vector store for semantic matches, with optional metadata filters.
        """
        # Build ChromaDB specific filter mapping
        where_clause = {}
        filters = []
        
        if section_filter:
            filters.append({"section": section_filter})
        if company_filter:
            filters.append({"company": company_filter})

        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}
        else:
            where_clause = None

        # Execute query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )

        # Parse and format results cleanly
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for d, m, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": d,
                    "metadata": m,
                    "similarity": 1.0 - dist # Cosine similarity = 1 - cosine distance
                })

        return formatted_results
