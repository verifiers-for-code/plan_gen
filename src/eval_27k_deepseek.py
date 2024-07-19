import os
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time

# Find the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_dir)

# Load environment variables from .env file
load_dotenv(os.path.join(root_dir, '.env'))

# Load the dataset
dataset = load_dataset("verifiers-for-code/failed_16_eval", split='test')

# Read the system prompt
with open(os.path.join(root_dir, 'prompts', 'system_prompts', 'deepseek.txt'), 'r') as file:
    system_prompt = file.read().strip()

# Read the format for the user prompt
with open(os.path.join(root_dir, 'prompts', 'format', 'deepseek.txt'), 'r') as file:
    user_prompt_format = file.read().strip()

# Initialize the DeepSeek client with API key from .env
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# Print the first prompt
# first_row = dataset[0]
# first_prompt = user_prompt_format.format(first_row['input'], first_row['code'])
# print("First prompt sent to DeepSeek API:")
# print(first_prompt)

def call_deepseek_api(user_prompt):
    max_retries = 1
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=1.0,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Returning None.")
                return None

def process_dataset(example):
    user_prompt = user_prompt_format.format(example['prompt'], example['canonical_solution'], example['cleaned-aurora_expt_16'], example['output'], example['failed_tests'])
    example['deepseek_eval'] = call_deepseek_api(user_prompt)
    return example

# Apply the processing function to each row and create a new column
dataset = dataset.map(process_dataset, desc="Processing rows")

# Push the updated dataset back to Hugging Face
dataset.push_to_hub("verifiers-for-code/failed_16_eval")
