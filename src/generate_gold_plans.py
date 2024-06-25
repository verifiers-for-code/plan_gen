import anthropic
from datasets import load_dataset
import os
from dotenv import load_dotenv

# Find the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_dir)

# Load environment variables from .env file
load_dotenv(os.path.join(root_dir, '.env'))

# Load the dataset
dataset = load_dataset("openai/openai_humaneval")

# Read the system prompt
with open(os.path.join(root_dir, 'prompts', 'system_prompts', 'sonnet_plangen.txt'), 'r') as file:
    system_prompt = file.read().strip()

# Read the format for the user prompt
with open(os.path.join(root_dir, 'prompts', 'format', 'sonnet_plangen.txt'), 'r') as file:
    user_prompt_format = file.read().strip()

# Initialize the Anthropic client with API key from .env
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def process_row(row):
    # Format the user prompt using the format from the file
    user_prompt = user_prompt_format.format(row['prompt'], row['canonical_solution'])
    
    # Call the Anthropic API
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        temperature=1.0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )
    
    # Extract only the text content from the response
    response_content = message.content
    if isinstance(response_content, list) and len(response_content) > 0:
        text_block = response_content[0]
        if hasattr(text_block, 'text'):
            return text_block.text
    
    # If we can't extract the text as expected, return the raw content
    return str(response_content)

# Apply the processing function to each row and create a new column
dataset = dataset.map(lambda row: {'sonnet-3.5_gold_plans': process_row(row)})

# Push the updated dataset back to Hugging Face
dataset.push_to_hub("verifiers-for-code/humaneval_plan_generation")