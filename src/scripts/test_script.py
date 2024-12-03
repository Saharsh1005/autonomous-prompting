import os
from tqdm import tqdm
from datasets import load_dataset
from typing import List
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')

from src.agents.planner import Planner
from src.agents.executor import Executor
from src.dataModel.gsm8k_data_model import GSM8KDataset, GSM8KDataRow
from src.scripts.llm_utils import extract_last_numeric_value, load_api_keys
from strategies.prompts import get_prompt


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


def process_item(item, planner, executor):
    """
    Process each item to generate the answer and collect necessary data.
    """
    question = item['question']
    correct_answer_text = item.get('answer')
    correct_answer = extract_last_numeric_value(correct_answer_text)

    reranked_results = planner.retrieve_and_rerank(question, top_k=5)
    prompt, strategy = planner.generate_prompt(question, reranked_results)

    # Execute the prompt using the Executor
    final_answer = executor.execute_prompt(prompt, strategy)

    data_row = GSM8KDataRow(
        question=question,
        generated_answer=final_answer,
        correct_answer=correct_answer,
        strategy=strategy,
        priority=None
    )

    return data_row

def create_gsm8k_test_set(model_version="meta/meta-llama-3-8b-instruct", save_path: str = None) -> GSM8KDataset:
    """
    Creates a test set from the GSM8K dataset and processes the last 25 rows.
    """
    # Load API Keys
    _, _, REPLICATE_API_TOKEN, COHERE_API_KEY, PINECONE_API_KEY = load_api_keys()
    REPLICATE_API_TOKEN = REPLICATE_API_TOKEN
    cohere_api_key = COHERE_API_KEY
    pinecone_api_key = PINECONE_API_KEY
    pinecone_env = "aws"
    index_name = "ap-retrieval"
    embedding_dim = 1024
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

    # Load GSM8K Dataset
    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_dataset = gsm8k_dataset['test']

    # Extract last 25 rows
    last_25_rows = test_dataset.select(range(len(test_dataset) - 25, len(test_dataset)))

    dataset = GSM8KDataset(data=[])  
    predictions = []
    correct_answers = []

    planner = Planner(cohere_api_key, pinecone_api_key, pinecone_env, index_name, embedding_dim)
    executor = Executor(model_name=model_version)

    for item in tqdm(last_25_rows, desc="Processing GSM8K Test Set"):
        
        result = process_item(item, planner, executor)
        dataset.data.append(result)
        
        # Collect predictions and correct answers for accuracy calculation
        predictions.append(result.generated_answer)
        correct_answers.append(result.correct_answer)

    import json
    import csv

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(dataset.model_dump(), f, indent=4)

        csv_path = save_path.replace('.json', '.csv')  

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "generated_answer", "correct_answer", "strategy", "priority"])
        writer.writeheader()  
        for data_row in dataset.data:

            data_dict = {
                "question": data_row.question,
                "generated_answer": data_row.generated_answer,
                "correct_answer": data_row.correct_answer,
                "strategy": data_row.strategy,
                "priority": data_row.priority,
            }

            writer.writerow(data_dict)

    # Calculate Accuracy
    print("Calculating accuracy...")
    accuracy = check_accuracy(predictions, correct_answers)

    return dataset, accuracy


if __name__ == "__main__":
    """
    Runs the GSM8K test set creation and accuracy check.
    """
    model_version = "meta/meta-llama-3-8b-instruct"
    dataset, accuracy = create_gsm8k_test_set(model_version=model_version, save_path="data/gsm8k_test_set.json")
    print(f"Test set created with {len(dataset.data)} entries.")
    print(f"Accuracy: {accuracy:.2f}%")