"""
Knowledge base service
"""
import os
import configure_data as config
import hashlib
from filelock import FileLock
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors
_VECTOR_SIZE = 384

def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )

_hash_lock = FileLock(config.md5_path + ".lock")

def get_string_hash(input_str, encoding='utf-8'):
    """Return a SHA-256 hex digest of the input string."""
    str_bytes = input_str.encode(encoding=encoding)
    return hashlib.sha256(str_bytes).hexdigest()

# Keep legacy name as an alias so nothing else breaks
get_string_md5 = get_string_hash

def is_duplicate_and_register(hash_str: str) -> bool:
    """Atomically check if hash_str is already recorded; if not, record it.
    Returns True if it was a duplicate (skip), False if newly registered.
    """
    with _hash_lock:
        if not os.path.exists(config.md5_path):
            open(config.md5_path, "w", encoding="utf-8").close()
        with open(config.md5_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == hash_str:
                    return True
        with open(config.md5_path, "a", encoding="utf-8") as f:
            f.write(hash_str + "\n")
        return False

    

class KnowledgeBaseService(object):
    def __init__(self):
        client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        _ensure_collection(client, config.collection_name)
        self.qdrant = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=FastEmbedEmbeddings(model_name=config.embedding_model_name),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,
            chunk_overlap = config.chunk_overlap,
            separators = config.separators,
            length_function = len,
        )

    def upload_by_str(self, data, filename):
        content_hash = get_string_hash(data)
        if is_duplicate_and_register(content_hash):
            return "skipped"
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {
            "source":filename,
            "create_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"admin"

        }
        self.qdrant.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        return "success"
    



 