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
MAX_TOKENS = 512
TOP_P = 1.0

# Detect root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# Read instruction prompt
INSTRUCTION_PROMPT_PATH = os.path.join(ROOT_DIR, "prompts", "system_prompts", "phi3_baseline.txt")

def read_text_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

instruction_prompt = read_text_file(INSTRUCTION_PROMPT_PATH)

llm = LLM(
    MODEL,
    # enable_prefix_caching=True,
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
    for row in data:
        user_content = row["prompt"]
        prompt = [
            {
                "role": "user",
                "content": instruction_prompt + "\n" + user_content,
            },
        ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts

def process_for_dataset_split(df, column_name="generated_phi3_baseline"):
    print("Generating Prompts...")
    prompts = make_prompts_for_data(df)
    
    # Print the first prompt
    print("First prompt being sent to the model:")
    print(prompts[0])
    
    print("Running Inference")
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    
    # Drop the column if it exists
    if column_name in df.column_names:
        df = df.remove_columns(column_name)
    
    df = df.add_column(column_name, generated_texts)
    return df

dataset = load_dataset(DATASET, split="test")
dataset = process_for_dataset_split(dataset)

# Save to disk and push to hub
try:
    dataset.save_to_disk(os.path.join(ROOT_DIR, "data"))
except Exception as e:
    print(f"Error saving to disk: {e}")

dataset.push_to_hub(DATASET)