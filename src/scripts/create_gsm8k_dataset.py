""" 
Author: Saharsh Barve, Ishaan Singh
Description: Script to preprocess & create the GSM8k dataset (4promptStrategy*1000 samples). It will store the final dataset in the data/ directory.
"""

import os
import re
import time
import random
import replicate
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pydantic import BaseModel
from typing import List, Literal, Union
from collections import Counter

# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')
# ======================================================================

from src.dataModel.gsm8k_data_model import GSM8KDataRow, GSM8KDataset
from src.strategies.prompts import get_prompt, zero_shot_prompt, few_shot_prompt, cot_prompt, sc_prompt
from src.scripts.llm_utils import extract_last_numeric_value, join_tokens, generate_llm_response, generate_self_consistent_answers, load_api_keys

def evaluate_strategies(df: pd.DataFrame, strategies: List[str], priority_map: dict) -> None:
    accuracy_per_strategy = {}
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]
        total = len(strategy_df)
        correct = 0
        for _, row in strategy_df.iterrows():
            try:
                if float(row['generated_answer']) == float(row['correct_answer']):
                    correct += 1
            except:
                pass
        accuracy = correct / total if total > 0 else 0
        accuracy_per_strategy[strategy] = accuracy * 100
        print(f"Strategy: {strategy}, Accuracy: {accuracy * 100:.2f}%")

    strategy_ranking = sorted(
        strategies,
        key=lambda x: (priority_map[x], -accuracy_per_strategy[x])
    )

    print("\nStrategy Ranking based on priority (lower is better) and performance:")
    for rank, strategy in enumerate(strategy_ranking, 1):
        print(f"{rank}. Strategy: {strategy}, Priority: {priority_map[strategy]}, Accuracy: {accuracy_per_strategy[strategy]:.2f}%")


def create_gsm8k_dataset(model_version = "meta/meta-llama-3-8b-instruct", SAMPLE_COUNT: int = 10, save_path: str = None) -> None:
    REPLICATE_API_TOKEN = load_api_keys()[2]
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_dataset = gsm8k_dataset['test']
    test_dataset = test_dataset.select(range(SAMPLE_COUNT))
    strategies = ["zero-shot", "few-shot", "cot", "sc-cot"]
    priority_map = {"zero-shot": 0, "few-shot": 1, "cot": 2, "sc-cot": 3}

    dataset = GSM8KDataset(data=[])

    

    for item in tqdm(test_dataset, desc="Processing GSM8K Dataset"):
        question = item['question']
        correct_answer_text = item.get('answer')

        correct_answer = extract_last_numeric_value(correct_answer_text)

        for strategy in strategies:
            prompt = get_prompt(strategy, question)

            if strategy == 'sc-cot':
                generated_answer = generate_self_consistent_answers(prompt, model_name=model_version, num_samples=7)
            else:
                response = generate_llm_response(prompt, model_name=model_version)
                generated_answer = extract_last_numeric_value(response)

            data_row = GSM8KDataRow(
                question=question,
                correct_answer=correct_answer,  
                generated_answer=generated_answer,  
                strategy=strategy,
                priority=priority_map[strategy]
            )
            dataset.data.append(data_row)

    data_records = [data_row.model_dump() for data_row in dataset.data]
    df = pd.DataFrame(data_records)
    df.to_csv(f"{save_path}.csv", index=False)
    df.to_json(f"{save_path}.json", orient='records', lines=True)

    evaluate_strategies(df, strategies, priority_map)
    print(f"Validated and saved dataset to {save_path}")

if __name__ == "__main__":
    os.makedirs('./data/', exist_ok=True)
    create_gsm8k_dataset(model_version = "meta/meta-llama-3-8b-instruct" ,SAMPLE_COUNT=1000,save_path='data/gsm8k_1k')




    
