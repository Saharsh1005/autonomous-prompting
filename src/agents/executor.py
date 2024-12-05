# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')
# ======================================================================
import re
import time
import replicate
from collections import Counter
import os

from planner import Planner
from scripts.llm_utils import extract_last_numeric_value, join_tokens, generate_llm_response, generate_self_consistent_answers, load_api_keys

class Executor:
    def __init__(self, model_name):
        """
        Initialize the Executor with the specified language model.

        Args:
        model_name (str): The name or path of the language model to use.
        """
        self.model_name = model_name

    def execute_prompt(self, prompt: str, strategy: str, num_samples: int = 5, system_prompt: str = None, debug: bool = False):
        """
        Execute the prompt using the specified strategy.

        Returns:
        str: The final generated answer.
        """
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        if strategy == 'sc-cot': # Use self-consistency, generate multiple samples and select the most common answer
            if debug:
                print(f"[Executor] Using self-consistency strategy to generate {num_samples} samples.")
            final_answer = generate_self_consistent_answers(prompt, self.model_name, num_samples)
        elif strategy == 'sp': # Use chain of thought reasoning, generate a single sample
            response = generate_llm_response(prompt, self.model_name, max_length=1024)
            final_answer = (response)
        else:
            if debug:
                print(f"[Executor] Using {strategy} strategy to generate a single sample.")
            response = generate_llm_response(prompt, self.model_name)
            if debug:
                print(f"[Executor] The Raw Response: {response}")
            final_answer = extract_last_numeric_value(response)
        return final_answer
    

# Example usage
if __name__ == "__main__":
    # Initialize API keys and model settings
    OPENAI_API_KEY, LLAMA_API_KEY, REPLICATE_API_TOKEN, COHERE_API_KEY, PINECONE_API_KEY = load_api_keys()
    REPLICATE_API_TOKEN = REPLICATE_API_TOKEN
    cohere_api_key = COHERE_API_KEY
    pinecone_api_key = PINECONE_API_KEY
    pinecone_env = "aws"
    index_name = "ap-retrieval"
    embedding_dim = 1024
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
    model_version = "meta/meta-llama-3-8b-instruct"
    new_question = "Rodney is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40perct of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Rodney has to restart the download from the beginning. How long does it take to download the file?"

    planner = Planner(cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim)
    executor = Executor(model_name=model_version)

    reranked_results = planner.retrieve_and_rerank(new_question, top_k=5)
    prompt, strategy = planner.generate_prompt(new_question, reranked_results)

    # Execute the prompt using the Executor
    final_answer = executor.execute_prompt(prompt, strategy)

    print(f"Final Answer: {final_answer}")
