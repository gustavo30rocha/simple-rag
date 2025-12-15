import argparse
import re
import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

CHROMA_PATH = "chroma"
BM25_INDEX_PATH = os.path.join(CHROMA_PATH, "bm25_index.pkl")
BM25_DOCS_PATH = os.path.join(CHROMA_PATH, "bm25_docs.pkl")

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context documents. Your answers must be grounded in the context provided below.

## Instructions:
1. Answer the question using ONLY information from the context provided
2. If the context contains the answer, provide a clear and accurate response
3. If multiple pieces of context are relevant, synthesize them into a coherent answer
4. If the context contradicts itself, acknowledge the contradiction and present both perspectives
5. If the context does NOT contain enough information to answer the question, explicitly state: "I don't have enough information in the provided documents to answer this question."
6. Do NOT use any knowledge outside of the provided context
7. Be specific and cite relevant details from the context when possible

## Context:
{context}

## Question:
{question}

## Answer:
"""

def _normalize(scores):
    """
    Normalize scores to 0-1 range using min-max normalization.
    Works with both lists and numpy arrays.
    
    Args:
        scores: List or numpy array of scores
    
    Returns:
        List of normalized scores in 0-1 range
    """
    # Convert to list if numpy array
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    if len(scores_list) == 0:
        return []
    
    smin = float(min(scores_list))
    smax = float(max(scores_list))
    
    # Handle identical scores (using threshold for floating-point precision)
    if smax - smin < 1e-12:
        return [0.0] * len(scores_list)
    
    # Min-max normalization
    return [(score - smin) / (smax - smin) for score in scores_list]

def distance_to_similarity(cosine_distance):
    """Convert cosine distance (0-2) to cosine similarity (-1 to 1)"""
    return 1 - cosine_distance

def tokenize(text):
    """Simple tokenizer for BM25: splits text into words"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def hybrid_search(db, query_text, k, hybrid_weight=0.7):
    """
    Perform hybrid search combining vector similarity and BM25 keyword search.
    Uses pre-built BM25 index from disk.
    
    Args:
        db: ChromaDB instance
        query_text: Query string
        k: Number of results to return
        hybrid_weight: Weight for vector search (0.0 = only BM25, 1.0 = only vector)
    
    Returns:
        List of (document, combined_score) tuples, sorted by score descending
    """
    # If hybrid_weight is 1.0 (or very close), do pure vector search
    if hybrid_weight >= 0.999:
        raw_results = db.similarity_search_with_score(query_text, k=k)
        # Convert distances to similarities, then normalize
        vector_distances = [score for _, score in raw_results]
        vector_similarities = [distance_to_similarity(dist) for dist in vector_distances]
        vector_scores_normalized = _normalize(vector_similarities)
        return [(doc, score) for (doc, _), score in zip(raw_results, vector_scores_normalized)]
    
    # Load BM25 index and document references
    if not os.path.exists(BM25_INDEX_PATH) or not os.path.exists(BM25_DOCS_PATH):
        print("BM25 index not found. Please run create_database.py first.")
        print("Falling back to vector search only.")
        raw_results = db.similarity_search_with_score(query_text, k=k)
        # Convert distances to similarities, then normalize
        vector_distances = [score for _, score in raw_results]
        vector_similarities = [distance_to_similarity(dist) for dist in vector_distances]
        vector_scores_normalized = _normalize(vector_similarities)
        return [(doc, score) for (doc, _), score in zip(raw_results, vector_scores_normalized)]
    
    #print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    
    with open(BM25_DOCS_PATH, 'rb') as f:
        all_docs = pickle.load(f)
    
    # Perform BM25 search
    query_tokens = tokenize(query_text)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores using min-max normalization
    bm25_scores_normalized = _normalize(bm25_scores)

    # Create BM25 score map by document content
    bm25_scores_map = {
        doc["content"]: score
        for doc, score in zip(all_docs, bm25_scores_normalized)
    }
    
    # Perform vector search (get more results to merge with BM25)
    vector_results = db.similarity_search_with_score(query_text, k=k * 3)
    
    # Convert cosine distances to similarities, then normalize
    vector_distances = [score for _, score in vector_results]
    vector_similarities = [distance_to_similarity(dist) for dist in vector_distances]
    vector_scores_normalized = _normalize(vector_similarities)
    
    # Create vector scores map
    vector_scores_map = {
        doc.page_content: score
        for doc, score in zip([doc for doc, _ in vector_results], vector_scores_normalized)
    }
    
    # Combine results: merge vector and BM25 scores
    combined_results = {}
    
    # Add all vector results
    for doc, normalized_score in zip([doc for doc, _ in vector_results], vector_scores_normalized):
        content = doc.page_content
        vector_score = normalized_score
        bm25_score = bm25_scores_map.get(content, 0.0)
        combined_score = (hybrid_weight * vector_score) + ((1 - hybrid_weight) * bm25_score)
        combined_results[content] = (doc, combined_score)
    
    # Add top BM25 results that might not be in vector results
    # Sort BM25 scores and get top k
    bm25_with_docs = list(zip(all_docs, bm25_scores_normalized))
    bm25_with_docs.sort(key=lambda x: x[1], reverse=True)
    
    for doc_data, bm25_score in bm25_with_docs[:k]:
        content = doc_data["content"]
        if content not in combined_results:
            doc = Document(
                page_content=content,
                metadata=doc_data["metadata"]
            )
            vector_score = vector_scores_map.get(content, 0.0)
            combined_score = (hybrid_weight * vector_score) + ((1 - hybrid_weight) * bm25_score)
            combined_results[content] = (doc, combined_score)
    
    # Convert to list and sort by combined score
    results = list(combined_results.values())
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return results[:k]

def query_rag(
    question: str,
    model: str = "llama3:8b",
    k: int = 5,
    hybrid: bool = False,
    hybrid_weight: float = 0.7,
    return_sources: bool = False
):
    """
    Query the RAG system and return the answer.
    
    Args:
        question: The question to ask
        model: Ollama model to use (default: llama3:8b)
        k: Number of documents to retrieve (default: 5)
        hybrid: Enable hybrid search (default: False)
        hybrid_weight: Weight for hybrid search, 0.0 = only BM25, 1.0 = only vector (default: 0.7)
        return_sources: If True, return (answer, sources) tuple; if False, return just answer (default: False)
    
    Returns:
        str or tuple: The answer text, or (answer, sources) if return_sources=True
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform search based on mode
    if hybrid:
        # Hybrid search: combine vector + keyword search
        results = hybrid_search(db, question, k, hybrid_weight)
    else:
        # Standard vector search
        raw_results = db.similarity_search_with_score(question, k=k)
        # Convert distances to similarities, then normalize
        vector_distances = [score for _, score in raw_results]
        vector_similarities = [distance_to_similarity(dist) for dist in vector_distances]
        vector_scores_normalized = _normalize(vector_similarities)
        results = [
            (doc, score)
            for (doc, _), score in zip(raw_results, vector_scores_normalized)
        ]

    # Check if there are any results
    if len(results) == 0:
        answer = "I don't have enough information in the provided documents to answer this question."
        if return_sources:
            return (answer, set())
        return answer
    
    # Format the retrieved documents as context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    model_instance = ChatOllama(model=model, temperature=0)
    response = model_instance.invoke(prompt)
    
    sources = [doc.metadata["source"] for doc, _score in results]
    sources = set(sources)  # Removes duplicates
    
    if return_sources:
        return (response.content, sources)
    return response.content
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3:8b", 
        help="Ollama model to use (default: llama3:8b)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (combines vector (dense) + keyword (sparse) search)"
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.7,
        help="Weight for hybrid search: 0.0 = only keyword, 1.0 = only vector (default: 0.7)"
    )
    args = parser.parse_args()

    answer, sources = query_rag(
        question=args.query_text,
        model=args.model,
        k=args.k,
        hybrid=args.hybrid,
        hybrid_weight=args.hybrid_weight,
        return_sources=True
    )
    
    formatted_response = f"Response: {answer}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()