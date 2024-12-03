
# ======== FixME: Ensure import works without src in PYTHONPATH ========
import json
import sys
sys.path.append('.')
sys.path.append('..')
# ======================================================================

from typing import Dict, List
from agents.planner import Planner
from agents.executor import Executor
from scripts.llm_utils import clean_numeric_value, extract_last_numeric_value_general, load_api_keys
from datasets import load_dataset

class ExperimentRunner:
    def __init__(self, dataset: str):
        '''
        Initialize the ExperimentRunner.
        '''
        self.dataset = dataset
        self.last_run_results = None

    def analyze_results(self, results: List[Dict], metric: str = 'accuracy'):
        '''
        Analyze the results of the experiment.
        '''
        if metric == 'accuracy':
            return sum(1 for entry in results if entry["answer"] == entry["ground_truth"]) / len(results)
        else:
            return None
        
    def save_results(self, results: List[Dict]):
        '''
        Save the results of the experiment.
        '''
        with open('results.json', 'w') as f:
            json.dump(results, f)

    def run(self, model_name: str = "meta/meta-llama-3-8b-instruct", top_k: int = 5):
        '''
        Run the experiment.
        '''
        cohere_api_key=load_api_keys()[3]
        pinecone_api_key=load_api_keys()[4]
        replicate_api_token=load_api_keys()[2]
        pinecone_env = "aws"
        index_name = "ap-retrieval"
        embedding_dim = 1024

        model_version = model_name

        planner = Planner(cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim)
        executor = Executor(model_name=model_version)
        results = []
        for index, datum in enumerate(self.dataset):
            print(f"Question {index+1}: {datum['question']}")
            new_question = datum['question']

            print(f"[ExperimentRunner] Reranking question {index+1}")
            reranked_results = planner.retrieve_and_rerank(new_question, top_k=top_k)
            prompt, strategy, cost = planner.generate_prompt(new_question, reranked_results)
            print(f"[ExperimentRunner] Generated prompt for question {index+1} with strategy: {strategy} with cost: {cost}")

            # Execute the prompt using the Executor
            print(f"[ExperimentRunner] Executing prompt for question {index+1}")
            final_answer = executor.execute_prompt(prompt, strategy)
            print(f"[ExperimentRunner] Final answer for question {index+1}: {final_answer} vs. {extract_last_numeric_value_general(datum['answer'])}")
            results.append({
                'question': new_question,
                'answer': clean_numeric_value(final_answer),
                'prompt': prompt,
                'strategy': strategy,
                'ground_truth': extract_last_numeric_value_general(datum['answer']),
                'cost': cost
            })
            print(f"[ExperimentRunner] Question {index+1} complete with results: {results[-1]}")
        self.last_run_results = results
        return results
    


if __name__ == "__main__":
    # Initialize API keys and Pinecone settings

    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset['test'].select(range(10))
    runner = ExperimentRunner(dataset=test_dataset)
    runner.run()
    print(runner.analyze_results(runner.last_run_results))
    print(runner.last_run_results)
    runner.save_results(runner.last_run_results)