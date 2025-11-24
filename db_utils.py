import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "gpsc_pilot_v1"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embeddings"

def get_embedding_function():
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBEDDING_MODEL
    )
    return ollama_ef

def get_chroma_collection(collection_name: str):
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    embed_fn = get_embedding_function()

    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)
    return collection