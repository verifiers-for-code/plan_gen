from datasets import load_dataset
import re

def extract_plan_content(text):
    # Use regex to find content between <plan> and </plan> tags
    match = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

def clean_dataset(example):
    example['cleaned_sonnet-3.5_gold_plans'] = extract_plan_content(example['sonnet-3.5_gold_plans'])
    return example

# Load the dataset
dataset = load_dataset("verifiers-for-code/humaneval_plan_generation")

# Apply the cleaning function to all splits
cleaned_dataset = dataset.map(clean_dataset)

# Push the updated dataset to the Hub
# Replace 'your-username' with your actual Hugging Face username
cleaned_dataset.push_to_hub("verifiers-for-code/humaneval_plan_generation")