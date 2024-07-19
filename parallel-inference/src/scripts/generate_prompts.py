from datasets import load_dataset
from transformers import AutoTokenizer
from utils import *
import torch
import json
from tqdm import tqdm

# HF Dataset
DATASET_NAME = "verifiers-for-code/Combined-Python-450k-filtered"
# Model
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_GPUS = 4
# Prompts
SYS_PROMPT = "prompts/system_prompts/plangen_27k.txt"
FORMAT_PROMPT = "prompts/formats/plangen_27k.txt"
FEW_SHOT_ROOT = "prompts/few_shot/plangen_27k/"
# JSON Output Location
OUTPUT_DIR = "data/inputs/"

def make_prompts_for_data(data):
    prompts = []
    for row in tqdm(data, desc="Creating prompts"):
        question = row["prompt"]
        answer = row["output"].strip()
        user_content = email_format.format(question, answer)
        prompt = prompt_base + [{"role": "user", "content": user_content}]
        prompts.append(prompt)
    prompts = prompts[:1000]
    prompts = [
        tokenizer.apply_chat_template(
            prompt, return_tensors=None, add_generation_prompt=True, tokenize=False
        )
        for prompt in tqdm(prompts, desc="Applying chat template")
    ]
    return prompts

def distribute_prompts(prompts, num_gpus):
    output = []
    total_prompts = len(prompts)
    distribution = total_prompts // num_gpus
    remainder = total_prompts % num_gpus
    
    start = 0
    for i in range(num_gpus):
        end = start + distribution + (1 if i < remainder else 0)
        output.append(prompts[start:end]) 
        start = end
    return output

def create_jsons(distributed_prompts, split_name):
    try:
        for i, prompts in enumerate(tqdm(distributed_prompts, desc=f"Creating JSON files for {split_name}")):
            location = f"{OUTPUT_DIR}prompts_{split_name}_{i}.json"
            with open(location, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False, indent=4)
            print(f"Prompts saved to {location}")
    except Exception as e:
        raise e

def process_split(dataset_split, split_name):
    print(f"Processing {split_name} split...")
    prompts = make_prompts_for_data(dataset_split)
    distributed_list = distribute_prompts(prompts, NUM_GPUS)
    print(f"Number of distributed lists: {len(distributed_list)}")
    print(f"Number of prompts in each list: {[len(sublist) for sublist in distributed_list]}")
    create_jsons(distributed_list, split_name)

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    email_format = read_text_file(FORMAT_PROMPT)

    prompt_base = [
        {"role": "system", "content": read_text_file(SYS_PROMPT)},
        {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_1_input.txt")},
        {
            "role": "assistant",
            "content": read_text_file(FEW_SHOT_ROOT + "example_1_output.txt"),
        },
        {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_2_input.txt")},
        {
            "role": "assistant",
            "content": read_text_file(FEW_SHOT_ROOT + "example_2_output.txt"),
        },
        {"role": "user", "content": read_text_file(FEW_SHOT_ROOT + "example_3_input.txt")},
        {
            "role": "assistant",
            "content": read_text_file(FEW_SHOT_ROOT + "example_3_output.txt"),
        },
    ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Process train split if it exists
    if 'train' in dataset:
        process_split(dataset['train'], 'train')

    # Process test split if it exists
    if 'test' in dataset:
        process_split(dataset['test'], 'test')

    print("Done!")