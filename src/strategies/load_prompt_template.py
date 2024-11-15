import yaml

def load_prompt_structure(file_path: str = 'src/strategies/prompt_templates.yaml'):
    """
    Load the YAML file containing prompt structures.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# ================== Preprocessing functions ==================
def zero_shot_preprocess(template: str, examples: list = None) -> str:
    """
    Preprocess the question for the zero-shot strategy.
    """
    return template

def few_shot_preprocess(template: str, examples: list = None) -> str:
    """
    Preprocess the question for the few-shot strategy.
    """
    if "{{#examples}}" in template:
        example_str = "\n\n".join(
            f"Q: {ex['input']}\nA: {ex['output']}" for ex in examples
        )
        template_ = template.replace("{{#examples}}", example_str)
    else:
        template_ = template

    return template_

def cot_preprocess(template: str, examples: list = None) -> str:
    """
    Preprocess the question for the COT strategy.
    """
    return template

def self_consistency_cot_preprocess(template: str, examples: list = None) -> str:
    """
    Preprocess the question for the self-consistency COT strategy.
    """
    return template
# ================================================================

def generate_prompt(strategy: str, question: str) -> str:
    """
    Generate a prompt based on the selected strategy.
    Returns: The formatted prompt.
    """
    # Load the prompt structure
    prompt_structure = load_prompt_structure()

    if strategy not in prompt_structure['prompts']:
        raise ValueError(f"Strategy '{strategy}' is not defined in the prompt structure.")
    
    strategy_config = prompt_structure['prompts'][strategy]
    template = strategy_config['template']
    examples = strategy_config.get('examples', [])
    
    # Get the preprocess function based on the strategy
    prompt = globals()[f"{strategy}_preprocess"](template, examples)
    
    # Replace the question placeholder with the actual question
    prompt = prompt.replace("{{question}}", question)
    
    return prompt

# Test
if __name__ == "__main__":
    strategies = ["zero_shot", "few_shot", "cot", "self_consistency_cot"]
    question = "What is the capital of Italy?"
    
    for i,strategy in enumerate(strategies):
        prompt = generate_prompt(strategy, question)
        print(f"{i}.Generated prompt for {strategy} strategy:\n\n{prompt}\n")