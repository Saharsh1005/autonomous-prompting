from openai import OpenAI
from transformers import pipeline,  AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import torch
import gc
import transformers
import re
# ================== Load API Keys ==================
def load_api_keys(model_name: str) -> str:
    """Load API keys from .env file."""
    load_dotenv()
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
    LLAMA_API_KEY       = os.getenv("HF_TOKEN")

    if model_name == "openai" and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing. Check your .env file.")
    if model_name == "meta" and not LLAMA_API_KEY:
        raise ValueError("LLAMA_API_KEY API key is missing. Check your .env file.")
    
    return OPENAI_API_KEY if model_name == "openai" else LLAMA_API_KEY
    

# ================== GPT-4 Model Call ==================
def generate_gpt_response(prompt,messages=None, model="gpt-4o-mini", temperature=0.7):
    if messages is None:
        messages = []
    
    messages.append({"role": "user", "content": prompt})

    openai_api_key = load_api_keys('openai')
    client=OpenAI()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = temperature
    )
    content=response.choices[0].message.content
    messages.append({'role': 'assistant', 'content': content})
    return {"response": content, "response_answer": extract_answer(content)}

def extract_answer(response):
    match = re.search(r"Final Numeric Answer:\s*[\*\(]*([\d\.]+)[\*\)]*", response)
    return match.group(1) if match else None

# ================== LLaMA Model Call [W.I.P - hasnt been tested]==================
def load_llama_model():
    """Load the LLaMA model and tokenizer with 4-bit quantization."""

    HF_TOKEN = load_api_keys("meta")
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Update as needed
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=HF_TOKEN,
    )

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
    )

    # Create a text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Free unused memory
    gc.collect()
    torch.cuda.empty_cache()

    return generator


# ================== LLaMA Model Call ==================
def generate_llama_response(prompt, max_length=512, temperature=0.7):
    """Generate a response using the LLaMA model."""
    try:
        generator = load_llama_model()
        response = generator(prompt, max_length=max_length, num_return_sequences=1, temperature=temperature)[0]
       
        return {"response": response['generated_text'], "response_answer": extract_answer(response['generated_text'])}
    except Exception as e:
        
        print(f"Error generating response with LLaMA: {e}")
        return "Error"    
        

if __name__ == "__main__":
    test_prompt = "What is 2 + 2?"
    # print("GPT-4 Response:", generate_gpt_response(test_prompt))
    print("LLaMA Response:", generate_llama_response(test_prompt))
