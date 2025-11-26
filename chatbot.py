import os, sys
import ollama
from typing import List, Dict, Any
from db_utils import get_chroma_collection, COLLECTION_NAME
from sentence_transformers import CrossEncoder
import torch

# System Prompt
SYSTEM_PROMPT = """
You are an intelligent assistant for the Graduate and Professional Student Council (GPSC).
Answer the user's question based ONLY on the provided context snippets from meeting minutes.
The context provided includes dates in the format [Date: YYYY-MM-DD].
If the context contains information from different academic years, clearly distinguish between them in your answer.

Your Goal: Answer the user's question accurately based on the context.

Structure your response into two distinct sections:
1. **Current Academic Year (2025-2026)**: Focus on events, decisions, and discussions from August 2025 to Present.
2. **Historical Context**: Summarize relevant information from previous academic years (e.g., Spring 2025, 2024, etc.) if available.

If the context does not contain the answer, politely state that you cannot find the information in the records.
Do not make up facts.
"""

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading ReRanker on: {device}")

RERANKER = CrossEncoder('BAAI/bge-reranker-base', device=device)

# Retrieval Layer
def retrieve_documents(query: str, n_results: int = 5, candidate_count: int = 25) -> List[Dict[str, Any]]:
    collection = get_chroma_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[query], n_results=candidate_count)
    return_list = []
    if results['documents'] and results['documents'][0]:
        reranker_list = [[query, document] for document in results["documents"][0]]
        relevance_scores = RERANKER.predict(reranker_list)
        zipped_tuple = zip(relevance_scores, results["documents"][0], results["metadatas"][0])
        sorted_list = sorted(zipped_tuple, reverse=True)[:n_results]
        # print(sorted(relevance_scores, reverse=True)[:1])
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
        cleaned_contexts = [f"[Date: {item['date']}] {item['text'].replace('\n', ' ')}" for item in contexts]
        context_str = "\n\n".join(cleaned_contexts)
        user_payload = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{user_question}\n\nINSTRUCTION:\nRemember to strictly structure your answer into two distinct sections: 'Current Academic Year (2025-2026)' and 'Historical Context'."
        response = ollama.chat(
            model="qwen2.5:32b",
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': user_payload
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