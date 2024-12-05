
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

class AutoPromptAgent:
    def __init__(self, cohere_api_key: str, pinecone_api_key: str, replicate_api_token: str, top_k: int = 5):
        '''
        Initialize the AutoPromptAgent.
        '''
        self.top_k = top_k
        self.cohere_api_key=cohere_api_key
        self.pinecone_api_key=pinecone_api_key
        self.replicate_api_token=replicate_api_token
        self.pinecone_env = "aws"
        self.index_name = "ap-retrieval"
        self.embedding_dim = 1024


        self.model_version = "meta/meta-llama-3-8b-instruct"
        self.planner = Planner(self.cohere_api_key, self.pinecone_api_key, self.pinecone_env, self.index_name, self.embedding_dim)
        self.executor = Executor(model_name=self.model_version)

    class SolutionPlannerAgent:
        def __init__(self, planner: Planner, executor: Executor, top_k: int = 5):
            self.planner = planner
            self.executor = executor
            self.top_k = top_k
            self.system_prompt = '''
            You are an intelligent problem-solving assistant. Your task is to help users solve complex problems effectively by using a two-step process:

            1. First, call the `get_best_strategy` function to determine the best strategy for solving the problem. This function takes the user's question as input and returns the optimal strategy and a reformulated prompt tailored to solving the problem.

            2. Once you receive the best strategy and the new prompt from the `get_best_strategy` function, proceed to solve the problem using the selected strategy. Use any relevant knowledge or reasoning to generate an accurate and detailed solution.

            Follow this workflow strictly:
            - If a user asks a question, first call the `get_best_strategy` function.
            - Use the returned prompt and strategy to proceed with solving the problem.
            - Only provide the final solution after completing both steps.

            If you need clarification or further inputs from the user, explicitly ask for them.

            You have access to the following functions:
            1. **get_best_strategy**:
            - Description: Determines the best strategy to solve a given problem and generates a tailored prompt.
            - Parameters:
                - `query` (string): The question or problem statement provided by the user.
            - Output: A JSON object containing:
                - `strategy` (string): The best strategy for solving the problem.
                - `prompt` (string): The reformulated prompt to be used with the chosen strategy.


            Always follow the workflow:
            1. Call `get_best_strategy` to get the strategy and new prompt.
            2. Use the returned prompt and strategy to solve the problem and provide the final answer.
        '''

        def get_best_strategy(self, question: str, debug: bool = False):
            '''
            Get the best strategy for solving the problem.
            '''
            reranked_results = self.planner.retrieve_and_rerank(question, top_k=self.top_k, debug=debug)
            prompt, strategy, cost = self.planner.generate_prompt(question, reranked_results, debug=debug)
            return prompt, strategy
        
        def function_mapping_callable(self, response: dict):
            '''
            Call the function with the given name and arguments.
            '''
            # Map the functions
            function_map = {
                "get_best_strategy": self.get_best_strategy,
            }

            if "function_call" in response["choices"][0]["message"]:
                function_call = response["choices"][0]["message"]["function_call"]
                function_name = function_call["name"]
                function_arguments = eval(function_call["arguments"])  # Convert stringified JSON to dict

                # Call the appropriate function
                result = function_map[function_name](**function_arguments)
                return (function_name, result)  
            return None, None
        
        def run(self, prompt: str, debug: bool = False):
            '''
            Plan the solution to the problem.
            '''
            # response = self.executor.execute_prompt(prompt, 'sp', system_prompt=self.system_prompt)
            # # print(response)
            # if 'function_call' in response: 
            #     prompt, strategy, cost = self.get_best_strategy(prompt)
            #     return prompt, strategy
            # else:
            #     return None, None
            return self.get_best_strategy(prompt, debug=debug)
            
        
    class SolutionExecutorAgent:
        def __init__(self, executor: Executor):
            self.executor = executor

        def run(self, prompt: str, strategy: str, debug: bool = False):
            '''
            Execute the prompt.
            '''
            return self.executor.execute_prompt(prompt, strategy, debug=debug)


    def run(self, question: str, debug: bool = False):
        '''
        Run the agent with the given question.
        '''
        functions = [
            {
                "name": "get_best_strategy",
                "description": "Determines the best strategy to solve a given problem and generates a tailored prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question or problem statement provided by the user."}
                    },
                    "required": ["query"]
                }
            },
        ]
        solution_planner_agent = self.SolutionPlannerAgent(self.planner, self.executor, self.top_k)
        solution_executor_agent = self.SolutionExecutorAgent(self.executor)
        prompt, strategy = solution_planner_agent.run(question, debug)
        if prompt is None or strategy is None:
            raise ValueError("Solution planner agent failed to return a prompt and strategy")
        answer = solution_executor_agent.run(prompt, strategy, debug)
        if debug:
            print(f"[AutoPromptAgent] The Final Answer: {answer} with strategy: {strategy}")
        return answer

    # def run(self, model_name: str = "meta/meta-llama-3-8b-instruct", top_k: int = 5, target_against_strategy: str = None, max_questions: int = None):
    #     '''
    #     Run the experiment.
    #     '''
    #     cohere_api_key=load_api_keys()[3]
    #     pinecone_api_key=load_api_keys()[4]
    #     replicate_api_token=load_api_keys()[2]
    #     pinecone_env = "aws"
    #     index_name = "ap-retrieval"
    #     embedding_dim = 1024

    #     model_version = '"meta/meta-llama-3-8b-instruct"'

    #     planner = Planner(cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim)
    #     executor = Executor(model_name=model_version)
    #     results = []

    #     count = 0
    #     for index, datum in enumerate(self.dataset):
    #         if max_questions and count >= max_questions:
    #             break
    #         print(f"Question {index+1}: {datum['question']}")
    #         new_question = datum['question']

    #         print(f"[ExperimentRunner] Reranking question {index+1}")
    #         reranked_results = planner.retrieve_and_rerank(new_question, top_k=top_k)
    #         if target_against_strategy:
    #             against_reranked_results = sorted(reranked_results, key=lambda x: x['strategy'] == target_against_strategy, reverse=True)
    #             if against_reranked_results[0]['strategy'] != target_against_strategy:
    #                 print(f"[ExperimentRunner] Skipping question {index+1} because the against strategy ({target_against_strategy}) is not present")
    #                 continue

    #         prompt, strategy, cost = planner.generate_prompt(new_question, reranked_results)
    #         print(f"[ExperimentRunner] Generated prompt for question {index+1} with strategy: {strategy} with cost: {cost}")

    #         # Execute the prompt using the Executor
    #         print(f"[ExperimentRunner] Executing prompt for question {index+1}")
    #         final_answer = executor.execute_prompt(prompt, strategy)
    #         print(f"[ExperimentRunner] Final answer for question {index+1}: {final_answer} vs. {extract_last_numeric_value_general(datum['answer'])}")
    #         if target_against_strategy:
    #             against_prompt, against_strategy, against_cost = planner.generate_prompt(new_question, against_reranked_results)
    #             print(f"[ExperimentRunner] Generated against prompt for question {index+1} with strategy: {against_strategy} with cost: {against_cost}")
    #             print(f"[ExperimentRunner] Against strategy: {against_strategy}")
    #             against_answer = executor.execute_prompt(against_prompt, against_strategy)
    #             print(f"[ExperimentRunner] Against answer for question {index+1}: {against_answer} vs. {extract_last_numeric_value_general(datum['answer'])}")

    #         results.append({
    #             'question': new_question,
    #             'answer': clean_numeric_value(final_answer),
    #             'prompt': prompt,
    #             'strategy': strategy,
    #             'against_prompt': against_prompt,
    #             'against_strategy': against_strategy,
    #             'against_answer': clean_numeric_value(against_answer),
    #             'ground_truth': extract_last_numeric_value_general(datum['answer']),
    #             'cost': cost,
    #             'against_cost': against_cost
    #         })
    #         print(f"[ExperimentRunner] Question {index+1} complete with results: {results[-1]}")
    #         count += 1
    #     self.last_run_results = results
        # return results



if __name__ == "__main__":
    # Initialize API keys and Pinecone settings

    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset['test']
    runner = AutoPromptAgent()
    against_strategy = 'cot'
    runner.run(target_against_strategy=against_strategy, max_questions=10)
    print(f"Accuracy (auto-prompt vs. {against_strategy}): ", runner.analyze_results(runner.last_run_results, 'accuracy', against_strategy))
    runner.save_results(runner.last_run_results)
