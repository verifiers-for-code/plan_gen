from datasets import load_dataset
import json
import random

SYS_PROMPT = """You are given the start of a function for a Python program. Your job is to produce a detailed plan. First, analyze and think about the function, then produce a plan. Do not generate any code. The function and docstring will be provided, so they do not need to be defined or initialized again within your plan.

Respond in the following format:

<thinking>
Your thought process and analysis of the function goes here. This should include considerations about the function's purpose, inputs, outputs, and any potential challenges or considerations.
</thinking>

<plan>
Your detailed plan for implementing the function goes here. This should outline the steps to implement the function without including actual code.
</plan>

Ensure your response follows this exact format, with the analysis enclosed in <thinking> tags and the plan enclosed in <plan> tags. The content within each tag should be a continuous paragraph without line breaks."""

def main():
    # Load the dataset
    hf_dataset = "verifiers-for-code/merged_50k"
    df = load_dataset(hf_dataset)

    print(f"Dataset loaded: {df}")

    # Get the total number of items in the "train" split
    total_items = len(df["train"])

    # Determine the sample size (min of 1000 or total items)
    # sample_size = min(1321312312, total_items)

    # Generate a list of random indices for sampling
    # sampled_indices = random.sample(range(total_items), sample_size)

    # Create a list to store sampled items
    sampled_items = []

    # Iterate through the sampled indices
    for index in range(total_items):
        item = df["train"][index]
        json_item = {
            "instruction": SYS_PROMPT,
            "input": item["input"],
            "output": item["gpt-4o-mini-fs"],
        }
        sampled_items.append(json_item)

    # Open a file to write the JSON output
    with open("../data/codegen50K_mixture.json", "w") as f:
        # Write the entire list as a single JSON array
        json.dump(sampled_items, f, indent=2)

    print(f"Conversion complete. written to ../data/codegen1k_sample_10.json")

if __name__ == "__main__":
    main()
