""" 
Author: Saharsh Barve
Description: Script to preprocess & create the GSM8k dataset (4promptStrategy*1000 samples). It will store the final dataset in the data/ directory.
"""

import pandas as pd
import random
from datasets import load_dataset
import json
import os

# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('src')
# ======================================================================

from src.dataModel.gsm8k_data_model import GSM8KDataRow, GSM8KDataset
from src.strategies.load_prompt_template import generate_prompt

def load_gsm8k_data(SAMPLE_COUNT: int  = 1000) -> pd.DataFrame:
    random.seed(42)
    ds = load_dataset("openai/gsm8k", "main")
    train_data = list(ds['train'])
    sampled_questions = random.sample(train_data, SAMPLE_COUNT)

    df = pd.DataFrame(sampled_questions)

    df['question'] = df['question'].str.strip().str.lower()
    df['answer'] = df['answer'].str.strip().str.lower()

    return df[['question', 'answer']]

def create_gsm8k_dataset(save_path: str = None) -> None:

    df = load_gsm8k_data()
    data_records = df.to_dict(orient='records')

    strategies = ["zero_shot", "few_shot", "cot", "self_consistency_cot"]
    
    dataset = GSM8KDataset(
        data=[
            GSM8KDataRow(
                **record,  
                prompt="",  
                strategy=_strategy  
            )
            for record in data_records
            for _strategy in strategies
        ]
    )  
    # ===== Add logic to populate prompt and strategy fields ===== 
    for data_row in dataset.data:
        question = data_row.question
        data_row.prompt = generate_prompt(data_row.strategy, question)
    # ============================================================
    
    # Save as JSON
    with open(save_path, "w") as f:
        f.write(dataset.model_dump_json(indent=4))
    
    print(f"Validated and saved dataset to {save_path}")

if __name__ == "__main__":
    os.makedirs('./data/', exist_ok=True)
    create_gsm8k_dataset(save_path='data/gsm8k_1k.json')




    
