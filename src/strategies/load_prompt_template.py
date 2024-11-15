import yaml

def load_prompt_structure(file_path: str = 'src/strategies/prompt_templates.yaml'):
    """
    Load the YAML file containing prompt structures.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

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
    
    # Handle the few-shot example substitution
    if "{{#examples}}" in template:
        example_str = "\n\n".join(
            f"Q: {ex['input']}\nA: {ex['output']}" for ex in examples
        )
        prompt = template.replace("{{#examples}}", example_str)
    else:
        prompt = template
    
    prompt = prompt.replace("{{question}}", question)
    
    return prompt

# Test
if __name__ == "__main__":
    strategies = ["zero_shot", "few_shot", "cot", "self_consistency_cot"]
    question = "What is the capital of Italy?"
    
    for i,strategy in enumerate(strategies):
        prompt = generate_prompt(strategy, question)
        print(f"{i}.Generated prompt for {strategy} strategy:\n{prompt}\n")