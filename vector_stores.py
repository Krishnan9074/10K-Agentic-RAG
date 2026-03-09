import configure_data as config
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

_VECTOR_SIZE = 384

def _ensure_collection(client, collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        client = config.make_qdrant_client()
        _ensure_collection(client, config.collection_name)
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=self.embedding,
        )

    def get_retriever(self):
        "Vector retriever for use in chains"
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
