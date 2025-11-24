import os, sys
import ollama
from typing import List, Dict, Any
from db_utils import get_chroma_collection, COLLECTION_NAME

# System Prompt
SYSTEM_PROMPT = """
You are an intelligent assistant for the Graduate and Professional Student Council (GPSC).
Answer the user's question based ONLY on the provided context snippets from meeting minutes.
If the context does not contain the answer, politely state that you cannot find the information in the records.
Do not make up facts.
"""

# Retrieval Layer
def retrieve_documents(query: str, n_results: int = 5) -> List[str]:
    collection = get_chroma_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[query], n_results=n_results)

    if results['documents'] and results['documents'][0]:
        return list(results['documents'][0])
    else:
        return []
    

# RAG
def query_rag(user_question: str) -> Dict[str, Any]:
    contexts = retrieve_documents(query=user_question)
    if contexts:
        cleaned_contexts = [doc.replace("\n", " ") for doc in contexts]
        context_str = "\n\n".join(cleaned_contexts)
        prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}\n\nQUESTION:\n{user_question}"
        response = ollama.chat(
            model="llama3",
            messages=[{'role': 'user', 'content': prompt}]
        )
        return {"answer": response['message']['content'], "context": contexts}
    else:
        return {"answer": "I could not find any relevant meeting minutes.", "context": []}

