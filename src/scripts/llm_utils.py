from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import replicate
from collections import Counter
import re
# ================== Load API Keys ==================
def load_api_keys() -> str:
    """Load API keys from .env file."""
    load_dotenv()
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
    LLAMA_API_KEY       = os.getenv("HF_TOKEN")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    COHERE_API_KEY      = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY") 

    return OPENAI_API_KEY, LLAMA_API_KEY, REPLICATE_API_TOKEN, COHERE_API_KEY, PINECONE_API_KEY
    

# ================== Helper Functions ==================
def clean_and_normalize_answer(answer):
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s\$.]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer

def join_tokens(response):
    if isinstance(response, list):
        return ''.join(response)
    return response

def extract_numeric(text):
    match = re.search(r"Final Numeric Answer:\s*[\*\(]*([\d\.]+)[\*\)]*", text)
    return match.group(1) if match else None

def extract_last_numeric_value(text):
    match = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return match[-1].strip()
    else:
        return None

def clean_numeric_value(text): 
    return re.sub(r'[^\d.+-]', '', text)

def extract_last_numeric_value_general(text):
    match = re.findall(r'([-+]?(\d*|\d+(,\d+)*)\.\d+(,\d+)*)|([-+]?\d+(,\d+)*)', text)
    if match:
        sorted_match = sorted(list(match[-1]), key=lambda x: len(x))
        clean_final_answer = clean_numeric_value(sorted_match[-1].strip())
        return clean_final_answer
    else:
        return None

# ================== LLM Response Generation ==================
def generate_llm_response(prompt: str, model_name: str, retries: int = 3, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """
    Generate a response from the language model for a given prompt.

    Returns:
    str: The generated response text.
    """
    for attempt in range(retries):
        try:
            response = replicate.run(
                model_name,
                input={"prompt": prompt, "max_length": max_length, "temperature": temperature, "top_p": top_p}
            )
            response_text = join_tokens(response)
            return response_text
        except Exception as e:
            print(f"Error on attempt {attempt+1}/{retries}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("Max retries reached, returning empty response.")
                return ""

def generate_self_consistent_answers(prompt: str, model_name: str, num_samples: int = 5, retries: int = 3):
    """
    Generate multiple answers for self-consistency and select the most common one.

    Returns:
    str: The most common answer among the generated samples.
    """
    answers = []
    for i in range(num_samples):
        for attempt in range(retries):
            try:
                response = replicate.run(
                    model_name,
                    input={"prompt": prompt, "max_length": 512, "temperature": 0.7, "top_p": 0.9}
                )
                response_text = join_tokens(response)
                answer = extract_last_numeric_value(response_text)
                if answer:
                    answers.append(answer)
                break
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{retries}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    print("Max retries reached, skipping this sample.")
                    continue
    if answers:
        answer_counts = Counter(answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        return most_common_answer
    else:
        return ""
        

if __name__ == "__main__":
    test_prompt = "What is 2 + 2?"
    # print("GPT-4 Response:", generate_gpt_response(test_prompt))
    # print("LLaMA Response:", generate_llama_response(test_prompt))
