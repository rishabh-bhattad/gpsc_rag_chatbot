import os, sys
import ollama
from typing import List, Dict, Any
from db_utils import get_chroma_collection, COLLECTION_NAME
from sentence_transformers import CrossEncoder

# System Prompt
SYSTEM_PROMPT = """
You are an intelligent assistant for the Graduate and Professional Student Council (GPSC).
Answer the user's question based ONLY on the provided context snippets from meeting minutes.
If the context does not contain the answer, politely state that you cannot find the information in the records.
Do not make up facts.
"""

RERANKER = CrossEncoder('BAAI/bge-reranker-base')

# Retrieval Layer
def retrieve_documents(query: str, n_results: int = 10, candidate_count: int = 50) -> List[Dict[str, Any]]:
    collection = get_chroma_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[query], n_results=candidate_count)
    return_list = []
    if results['documents'] and results['documents'][0]:
        reranker_list = [[query, document] for document in results["documents"][0]]
        relevance_scores = RERANKER.predict(reranker_list)
        zipped_tuple = zip(relevance_scores, results["documents"][0], results["metadatas"][0])
        sorted_list = sorted(zipped_tuple, reverse=True)[:n_results]
        print(sorted(relevance_scores)[:1])
        for scores, document, metadata in sorted_list:
            return_list.append(
                {
                    "text": document.replace("\n", " ").replace("\u200b", ""),
                    "source": metadata['source'],
                    "date": metadata['date']
                }
            )
    return return_list
    

# RAG
def query_rag(user_question: str) -> Dict[str, Any]:
    contexts = retrieve_documents(query=user_question)
    if contexts:
        cleaned_contexts = [dict['text'].replace("\n", " ") for dict in contexts]
        context_str = "\n\n".join(cleaned_contexts)
        prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}\n\nQUESTION:\n{user_question}"
        response = ollama.chat(
            model="qwen2.5:32b",
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return {"answer": response['message']['content'], "context": contexts}
    else:
        return {"answer": "I could not find any relevant meeting minutes.", "context": []}


if __name__ == "__main__":
    while True:
        user_query = input("Ask me something: ")
        if user_query in ["quit", "exit"]:
            break
        response = query_rag(user_question=user_query)
        print(f"\nAI: {response['answer']}")
        print("\nSources:")
        if response['context']:
            seen_sources = []
            for item in response['context']:
                if item['source'] not in seen_sources:
                    print(f"File: {item['source']} | Date: {item['date']}")
                    seen_sources.append(item['source'])
        else:
            print("No sources used.")
            
        print("\n" + "-"*30 + "\n")