import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context documents.

Use the following pieces of context to answer the question. If you don't know the answer based on the context alone, say that you don't have enough information in the provided documents to answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise, and accurate answer based solely on the context provided above. If the context contains multiple relevant pieces of information, synthesize them into a coherent response. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
"""

def main():
    # Create CLI.
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
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB - must use the same embedding model as create_database.py
    # Using BAAI/bge-small-en-v1.5 - better quality than MiniLM, same size
    # Trained specifically for retrieval tasks, better semantic understanding
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB and retrieve relevant chunks with scores
    raw_results = db.similarity_search_with_score(query_text, k=args.k)

    # Convert cosine distance to cosine similarity, then normalize
    def distance_to_similarity(cosine_distance):
        """Convert cosine distance (0-2) to cosine similarity (-1 to 1), then normalize to (0-1)"""
        cosine_similarity = 1 - cosine_distance  # Convert distance to similarity
        normalized = (cosine_similarity + 1) / 2  # Normalize to 0-1
        return normalized

    results = [
        (doc, distance_to_similarity(distance)) # distance == raw_score
        for doc, distance in raw_results
    ]

    # Check if we got any results
    if len(results) == 0:
        return
    
    top_score = results[0][1]
    
    print(f"Top result score: {top_score:.4f}")

    # Format the retrieved documents as context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #print(f"Context: {context_text}")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context= context_text, question= query_text)
    #print(f"Prompt: {prompt}")

    model = ChatOllama(model=args.model, temperature=0)
    response = model.invoke(prompt)
    sources = [doc.metadata["source"] for doc, _score in results]
    formatted_response = f"Response: {response.content}\nSources: {sources}"
    
    print(formatted_response)

if __name__ == "__main__":
    main()