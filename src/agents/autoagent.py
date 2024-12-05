
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
                "description": "Deter