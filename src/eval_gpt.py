import os
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

# Find the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_dir)

# Load environment variables from .env file
load_dotenv(os.path.join(root_dir, '.env'))

# Load the dataset
dataset = load_dataset("verifiers-for-code/failed_16_eval")

# Read the system prompt
with open(os.path.join(root_dir, 'prompts', 'system_prompts', 'deepseek.txt'), 'r') as file:
    system_prompt = file.read().strip()

# Read the format for the user prompt
with open(os.path.join(root_dir, 'prompts', 'format', 'deepseek.txt'), 'r') as file:
    user_prompt_format = file.read().strip()

# Initialize the OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def process_row(row):
    # Format the user prompt using the format from the file
    user_prompt = user_prompt_format.format(row['prompt'], row['canonical_solution'], row['cleaned-aurora_expt_16'], row['output'], row['failed_tests'])
    
    # Call the OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=1.0
    )
    
    # Extract the response content
    return completion.choices[0].message.content

# Apply the processing function to each row and create a new column
dataset = dataset.map(lambda row: {'gpt-4-1106-preview_eval': process_row(row)})

# Push the updated dataset back to Hugging Face
dataset.push_to_hub("verifiers-for-code/failed_16_eval")