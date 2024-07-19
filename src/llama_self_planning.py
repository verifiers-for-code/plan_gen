import sys
import os
import json
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Constants
DATASET = "verifiers-for-code/humaneval_plan_generation"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_UTILIZATION = 0.93
TEMPERATURE = 1.0
MAX_TOKENS = 1024
TOP_P = 0.95
PROMPT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prompts")
)
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SYS_PROMPT = PROMPT_ROOT + "/system_prompts/self_planning.txt"
FEW_SHOT_ROOT = PROMPT_ROOT + "/few_shot/self_planning/"
EMAIL_FORMAT = PROMPT_ROOT + "/format/self_planning.txt"

def read_text_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

email_format = read_text_file(EMAIL_FORMAT)
system_prompt = read_text_file(SYS_PROMPT)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_1_input.txt")},
    {"role": "assistant", "content": read_text_file(FEW_SHOT_ROOT + "example_1_output.txt")},
    {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_2_input.txt")},
    {"role": "assistant", "content": read_text_file(FEW_SHOT_ROOT + "example_2_output.txt")},
    {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_3_input.txt")},
    {"role": "assistant", "content": read_text_file(FEW_SHOT_ROOT + "example_3_output.txt")},
]

llm = LLM(
    MODEL,
    enable_prefix_caching=True,
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
        input = row["prompt"]
        user_content = email_format.format(input)
        prompt = messages + [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts

def write_outputs_to_json_file(outputs, filename):
    try:
        with open(filename, "w") as file:
            json.dump(outputs, file, indent=4)
        print(f"Successfully wrote outputs to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def process_for_dataset_split(df, column_name="self_planning_" + MODEL.split("/")[-1]):
    print("Generating Prompts...")
    prompts = make_prompts_for_data(df)
    
    # Print the first prompt for verification
    print("\nFirst Prompt for Verification:")
    print(prompts[0])
    print("\n")
    
    print("Running Inference")
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    print(len(generated_texts))
    write_outputs_to_json_file(generated_texts, FEW_SHOT_ROOT + "outputs.json")
    df = df.add_column(column_name, generated_texts)
    return df

# Load and process dataset
dataset = load_dataset(DATASET, split="test")
dataset = process_for_dataset_split(dataset)

# Save and push to hub
try:
    dataset.save_to_disk(DATA_ROOT)
except:
    pass
dataset.push_to_hub(DATASET)