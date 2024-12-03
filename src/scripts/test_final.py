import time
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
import json
# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')
# ======================================================================
from src.agents.planner import Planner
from src.agents.executor import Executor
from src.scripts.llm_utils import extract_last_numeric_value, load_api_keys
from src.strategies.prompts import get_prompt

def check_accuracy(predictions: list, correct_answers: list) -> float:
    """
    Computes the accuracy of predictions compared to correct answers.
    """
    if len(predictions) != len(correct_answers):
        raise ValueError("Predictions and correct answers must have the same length.")
    
    correct = sum(
        str(pred).strip() == str(correct).strip()
        for pred, correct in zip(predictions, correct_answers)
    )
    total = len(correct_answers)
    accuracy = correct / total * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def count_tokens(text: str, tokenizer) -> int:
    """
    Counts the number of tokens in the given text using a tokenizer.
    """
    return len(tokenizer.encode(text)) if tokenizer else len(text.split())


def save_to_json(data, path: str):
    """
    Saves a dataset to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def measure_runtime(func):
    """
    Decorator for measuring function runtime.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        runtime = time.time() - start_time
        return result, runtime
    return wrapper

@measure_runtime
def process_item(item, planner, executor, tokenizer=None):
    """
    Process each item to generate the answer and collect necessary data.
    """
    question = item['question']
    correct_answer_text = item.get('answer')
    correct_answer = extract_last_numeric_value(correct_answer_text)

    # Generate prompt and strategy
    reranked_results = planner.retrieve_and_rerank(question, top_k=5)
    prompt, strategy = planner.generate_prompt(question, reranked_results)

    # Measure token usage
    token_count = count_tokens(prompt, tokenizer)

    # Execute the prompt using the Executor
    final_answer = executor.execute_prompt(prompt, strategy)

    # Return as a dictionary
    return {
        "question": question,
        "generated_answer": final_answer,
        "correct_answer": correct_answer,
        "strategy": strategy,
        "token_usage": token_count
    }

from tqdm import tqdm
from datasets import load_dataset


def run_test(model_version, strategy, planner=None, tokenizer=None, num_samples=1, save_path=None):
    """
    Runs a test on the GSM8K dataset using a given strategy.
    """
    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_dataset = gsm8k_dataset['test']
    last_25_rows = test_dataset.select(range(len(test_dataset) - 25, len(test_dataset)))

    dataset = []
    predictions = []
    correct_answers = []
    total_tokens = 0
    total_runtime = 0.0

    executor = Executor(model_name=model_version)

    for item in tqdm(last_25_rows, desc=f"Processing GSM8K Test Set ({strategy})"):
        result, runtime = process_item(item, planner, executor, tokenizer)
        result["runtime"] = runtime
        dataset.append(result)

        # Collect predictions and correct answers for accuracy calculation
        predictions.append(result["generated_answer"])
        correct_answers.append(result["correct_answer"])
        total_tokens += result["token_usage"]
        total_runtime += runtime

    # Save dataset if save_path is provided
    if save_path:
        save_to_json(dataset, save_path)

    # Calculate accuracy
    accuracy = check_accuracy(predictions, correct_answers)

    # Print statistics
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Tokens per Question: {total_tokens / len(dataset):.2f}")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Average Runtime per Question: {total_runtime / len(dataset):.2f} seconds")

    return dataset, accuracy

def run_test(
    strategy: str,
    model_version: str,
    save_path: str = None,
    dataset_slice: int = 25,
    num_samples: int = 1,
    tokenizer=None
):
    """
    Runs a test with the specified strategy and saves results to a file if save_path is provided.
    """
    # Load API Keys
    _, _, REPLICATE_API_TOKEN, COHERE_API_KEY, PINECONE_API_KEY = load_api_keys()
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

    # Load GSM8K Dataset
    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_dataset = gsm8k_dataset['test']
    rows_to_test = test_dataset.select(range(len(test_dataset) - dataset_slice, len(test_dataset)))

    dataset = []
    predictions = []
    correct_answers = []
    total_tokens = 0
    total_runtime = 0.0

    planner = Planner(COHERE_API_KEY, PINECONE_API_KEY, "aws", "ap-retrieval", 1024) if strategy == 'autoprompt' else None
    executor = Executor(model_name=model_version)

    for item in tqdm(rows_to_test, desc=f"Processing GSM8K Test Set ({strategy.upper()})"):
        question = item["question"]
        correct_answer_text = item.get("answer")
        correct_answer = extract_last_numeric_value(correct_answer_text)

        if strategy == "autoprompt":
            reranked_results = planner.retrieve_and_rerank(question, top_k=5)
            prompt, strategy_type = planner.generate_prompt(question, reranked_results)
        else:
            prompt = get_prompt(strategy, question)
            strategy_type = strategy
        
        print('-'*20)
        print(f"Strategy: {strategy_type}")
        print(f"Prompt: {prompt}")
        print('-'*20)

        token_usage = count_tokens(prompt, tokenizer) if tokenizer else len(prompt.split())
        start_time = time.time()
        generated_answer = executor.execute_prompt(prompt, strategy_type, num_samples if strategy == "sc-cot" else 1)
        end_time = time.time()

        runtime = end_time - start_time
        total_tokens += token_usage
        total_runtime += runtime

        result = {
            "question": question,
            "prompt": prompt,
            "generated_answer": generated_answer,
            "correct_answer": correct_answer,
            "strategy": strategy_type,
            "token_usage": token_usage,
            "runtime": runtime
        }
        dataset.append(result)
        predictions.append(generated_answer)
        correct_answers.append(correct_answer)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(dataset, f, indent=4)

    accuracy = check_accuracy(predictions, correct_answers)
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Tokens per Question: {total_tokens / len(dataset):.2f}")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Average Runtime per Question: {total_runtime / len(dataset):.2f} seconds")

    return dataset, accuracy


def run_all_tests():
    model_version = "meta/meta-llama-3-8b-instruct"
    tokenizer = None  # Replace with your tokenizer if needed

    # AutoPrompt Test
    print("Running AutoPrompt Test...")
    run_test(strategy="autoprompt", model_version=model_version, save_path="autoprompt_results.json", tokenizer=tokenizer)

    # Chain-of-Thought (CoT) Test
    print("Running CoT Test...")
    run_test(strategy="cot", model_version=model_version, save_path="cot_results.json", tokenizer=tokenizer)

    # Self-Consistency CoT Test
    print("Running Self-Consistency CoT Test...")
    run_test(strategy="sc-cot", model_version=model_version, save_path="sc_cot_results.json", tokenizer=tokenizer, num_samples=5)


if __name__ == "__main__":
    run_all_tests()
