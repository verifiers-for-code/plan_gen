import sys
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch

# Constants
DATASET = "verifiers-for-code/humaneval_plan_generation"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_UTILIZATION = 0.93
TEMPERATURE = 0.0
MAX_TOKENS = 2048
TOP_P = 0.95
__MAGIC_SPLITTER__ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

# Detect root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

llm = LLM(
    MODEL,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    tensor_parallel_size=NUM_GPUS,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P
)

tokenizer = llm.get_tokenizer()

print("Loaded Model")

def make_prompts_for_data(data):
    prompts = []
    response = f"""
    Below is a self-contained Python script that solves the problem: 
    ```python 
    {__MAGIC_SPLITTER__}
    ```
    """
    for row in data:
        user_content = row["cleaned_sonnet-3.5_gold_plans"]
        prompt = f"Please provide a self-contained Python script that solves the following problem in a markdown code block. Follow the given plan.\n```\n{user_content.strip()}\n```\n"
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False
        ).split(__MAGIC_SPLITTER__)[0]
        prompts.append(chat_prompt)
    return prompts

def process_for_dataset_split(df, column_name="generated_phi3_plan_generation"):
    print("Generating Prompts...")
    prompts = make_prompts_for_data(df)
    
    # Print the first prompt
    print("First prompt being sent to the model:")
    print(prompts[0])
    
    print("Running Inference")
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    
    # Process the generated texts
    processed_texts = []
    for text in generated_texts:
        # Extract the code from the markdown block
        start = text.find("```python")
        end = text.rfind("```")
        if start != -1 and end != -1:
            code = text[start+9:end].strip()
        else:
            code = text  # If no markdown block found, use the entire text
        processed_texts.append(code)
    
    # Drop the column if it exists
    if column_name in df.column_names:
        df = df.remove_columns(column_name)
    
    df = df.add_column(column_name, processed_texts)
    return df

dataset = load_dataset(DATASET, split="test")
dataset = process_for_dataset_split(dataset)

# Save to disk and push to hub
try:
    dataset.save_to_disk(os.path.join(ROOT_DIR, "data"))
except Exception as e:
    print(f"Error saving to disk: {e}")

dataset.push_to_hub(DATASET)