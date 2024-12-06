import pandas as pd
from retriever import initialize_cohere, initialize_pinecone, store_embeddings, retrieve_top_k_questions
from reranker import rerank_results

def main():
    # Initialize API keys and environment
    cohere_api_key = "" #removed keys for protection
    pinecone_api_key = "" # removed keys for protection
    pinecone_env = "aws"
    index_name = "ap-retrieval"
    embedding_dim = 1024

    # Initialize clients
    cohere_client = initialize_cohere(cohere_api_key)
    index = initialize_pinecone(pinecone_api_key, pinecone_env, index_name, embedding_dim)

    # Load dataset
    csv_file_path = "data/gsm8k_generated_dataset.csv"
    dataset = pd.read_csv(csv_file_path)

    # Store embeddings (Uncomment if embeddings need to be updated)
    store_embeddings(index, dataset, cohere_client, batch_size=100)

    # Retrieve top-k questions
    new_question = "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?"
    results = retrieve_top_k_questions(index, cohere_client, new_question, top_k=5)

    # Print retrieved questions
    print("Retrieved Questions:")
    for question in results:
        print(f"Question: {question['question']}, "
              f"Strategy: {question['strategy']}, "
              f"Priority: {question['priority']}, "
              f"Score: {question['score']}")

    # Re-rank the results
    reranked_results = rerank_results(results)

    # Print re-ranked results
    print("\nReranked Results:")
    for strategy in reranked_results:
        print(f"Question: {strategy['question']}, "
              f"Strategy: {strategy['strategy']}, "
              f"Priority: {strategy['priority']}, "
              f"Cost: {strategy['cost']}, "
              f"Score: {strategy['score']}")

if __name__ == "__main__":
    main()
