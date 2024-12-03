# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')
# ======================================================================

from retrieval.retriever import initialize_cohere, initialize_pinecone, retrieve_top_k_questions
from retrieval.reranker import rerank_results
from strategies.prompts import get_prompt
from scripts.llm_utils import load_api_keys
class Planner:
    def __init__(self, cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim):
        """
        Initialize Planner with Cohere and Pinecone settings.

        Args:
        cohere_api_key (str): API key for Cohere.
        pinecone_api_key (str): API key for Pinecone.
        pinecone_env (str): Pinecone environment.
        index_name (str): Name of the Pinecone index.
        embedding_dim (int): Dimension of the embeddings.
        """
        self.cohere_client = initialize_cohere(cohere_api_key)
        self.index = initialize_pinecone(pinecone_api_key, pinecone_env, index_name, embedding_dim)

    def retrieve_and_rerank(self, new_question: str, top_k: int=5):
        """
        Retrieve top-K questions and rerank them.

        Returns:
        list of dict: Reranked results with metadata.
        """
        print(f"[Planner] Retrieving top {top_k} similar questions for: {new_question}")
        retrieved_questions = retrieve_top_k_questions(self.index, self.cohere_client, new_question, top_k)
        print(f"[Planner] Retrieved Questions:")
        for idx, item in enumerate(retrieved_questions, 1):
            print(f"  {idx}. {item['question']} (Strategy: {item['strategy']}, Priority: {item['priority']})")
        print(f'\n')

        reranked_results = rerank_results(retrieved_questions)
        print(f"[Planner] Reranked Results:")
        for idx, item in enumerate(reranked_results, 1):
            print(f"  {idx}. {item['question']} (Strategy: {item['strategy']}, Cost: {item['cost']})")
        print(f'\n')
        return reranked_results

    def generate_prompt(self, query: str, reranked_results: list):
        """
        Generate the final prompt for the Executor Agent using get_prompt.

        Returns:
        tuple: Generated prompt and selected strategy.
        """
        # Select the best strategy based on reranked results
        if reranked_results:
            best_strategy = reranked_results[0]['strategy']
            print(f"\n------\n[Planner] Selected best strategy: {best_strategy}\n-----\n")
        else:
            best_strategy = "zero-shot"
            print(f"[Planner] No reranked results found. Defaulting to strategy: {best_strategy}")

        # Generate the prompt using the selected strategy
        prompt = get_prompt(best_strategy, query)
        return prompt, best_strategy

# Example usage
if __name__ == "__main__":
    # Initialize API keys and Pinecone settings
    cohere_api_key = ""
    pinecone_api_key = ""
    cohere_api_key=load_api_keys()[3]
    pinecone_api_key=load_api_keys()[4]
    pinecone_env = "aws"
    index_name = "ap-retrieval"
    embedding_dim = 1024

    # Input question
    new_question = "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?"

    planner = Planner(cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim)

    reranked_results = planner.retrieve_and_rerank(new_question, top_k=5)
    prompt, strategy = planner.generate_prompt(new_question, reranked_results)