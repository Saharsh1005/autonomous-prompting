# Autonomous Prompt Optimization for Large Language Models (LLMs)

This project aims to build an autonomous system to optimize prompt generation for Large Language Models (LLMs), enhancing performance across tasks by balancing generality and specificity. By automating the creation and selection of diverse, high-quality prompts, this system seeks to reduce manual intervention and maximize LLM utility across various applications.

**Proposal Document**: [Link to Proposal](https://docs.google.com/document/d/1NuH-juFnK-06XQE0cOYiUpV2loC1j3ePd4-OfXM7r2A/edit)

---

## Project Outline

**Directory Structure**
```
autonomous-prompting/
├── .github/                # GitHub workflows and configurations
├── auto-cot/               # Code and resources for Chain-of-Thought and Auto-CoT models
├── data/                   # Raw and processed datasets
├── docs/                   # Project documentation and markdown guides
├── src/                    # Core code for prompt generation, retrieval, agents, etc.
│   ├── strategies/         # Code for each prompt strategy (e.g., Zero-Shot, Few-Shot, CoT)
│   │   ├── zero_shot.py
│   │   ├── few_shot.py
│   │   ├── cot.py
│   │   └── cot_self_consistency.py
│   ├── retrieval/          # Code for the retriever and re-ranker functions
│   │   ├── generate_embeddings.py
│   │   └── retrieve_rerank.py
│   ├── agents/             # Code for planner and executor agents
│   │   ├── planner.py
│   │   └── executor.py
│   └── data/               # Scripts for dataset preparation and loading
│       └── prepare_dataset.py
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── .gitmodules             # Git submodules, if any
├── LICENSE                 # License for the project
└── README.md               # Overview of the project

```

**Step 1: Initial Setup + Data Preparation** (@Saharsh1005)
- Task A: Define a prompt structure template for each strategy (Zero-Shot, Few-Shot, CoT, Auto-CoT) in src/strategies/prompt_structures.yaml
- Task B: Write a script (src/strategies/prompt_loader.py) to dynamically load and format prompt structures from the YAML file.

## Group members
- Ishaan Singh: is14@illinois.edu
- Henry Yi: weigang2@illinois.edu
- Saharsh Barve: ssbarve2@illinois.edu
- Veda Kailasam: vedak2@illinois.edu

--- 

**Contributions and Future Work**: This project will contribute to automating prompt engineering for LLMs, promoting scalability, and improving LLM performance across diverse applications. Future directions may include real-time prompt adjustment based on task complexity.
