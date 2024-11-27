""" 
Author: Saharsh Barve, Ishaan Singh
Description: Script to preprocess & create the GSM8k dataset (4promptStrategy*1000 samples). It will store the final dataset in the data/ directory.
"""

import pandas as pd
import random
from datasets import load_dataset
import json
import os
from typing import Literal
from tqdm import tqdm

# ======== FixME: Ensure import works without src in PYTHONPATH ========
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')
# ======================================================================

from src.dataModel.gsm8k_data_model import GSM8KDataRow, GSM8KDataset
from src.strategies.prompts import get_prompt
from models import generate_gpt_response, generate_llama_response

def load_gsm8k_data(SAMPLE_COUNT: int  = 1000) -> pd.DataFrame:
    random.seed(42)
    ds = load_dataset("openai/gsm8k", "main")
    train_data = list(ds['train'])
    sampled_questions = random.sample(train_data, SAMPLE_COUNT)

    df = pd.DataFrame(sampled_questions)

    df['question'] = df['question'].str.strip().str.lower()
    df['answer'] = df['answer'].str.strip().str.lower()

    return df[['question', 'answer']]

def create_gsm8k_dataset(model: Literal['openai','llama'], dataset_size: int = 3, save_path: str = None) -> None:

    df = load_gsm8k_data(SAMPLE_COUNT=dataset_size)
    data_records = df.to_dict(orient='records')

    strategies = ["zero-shot", "few-shot", "cot", "sc"]
    
    dataset = GSM8KDataset(
        data=[
            GSM8KDataRow(
                **record,  
                prompt="",  
                strategy=_strategy,
                response=""  
            )
            for record in data_records
            for _strategy in strategies
        ]
    )  
    # ===== get prompt and generate corresponding response ===== 
    for data_row in tqdm(dataset.data, desc="Generating prompts and responses"):
        question = data_row.question
        data_row.prompt = get_prompt(data_row.strategy, question)

        if model == 'llama':
            model_response = generate_llama_response(data_row.prompt) #FIXME: Check if this is correct
        else:
            model_response = generate_gpt_response(data_row.prompt)[1]['content']
        
        data_row.response = model_response

    # Save as JSON
    with open(save_path, "w") as f:
        f.write(dataset.model_dump_json(indent=4))
    
    print(f"Validated and saved dataset to {save_path}")

if __name__ == "__main__":
    os.makedirs('./data/', exist_ok=True)
    create_gsm8k_dataset(model='openai' ,dataset_size=1000 ,save_path='data/gsm8k_1k.json')




    
