import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import *
import torch
import json

# VLLM 
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_GPUS = 1
# JSON Output Location
DATA_INPUT = "data/inputs/"
DATA_OUTPUT = "data/outputs/"
SPLIT_NAMES = ["train"]

def read_json_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        content = json.load(f)
    return content

def create_jsons(split_name, index):
    try:
        location = f"{DATA_INPUT}prompts_{split_name}_{index}.json"
        content = read_json_file(location)
        print(f"Prompts loaded from {location}")
        return [content]
    except Exception as e:
        raise e

def dump_json_output(output, index, split_name):
    try:
        location = f"{DATA_OUTPUT}output_{split_name}_{index}.json"
        with open(location, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"Output saved to {location}")
    except Exception as e:
        raise e

def main(load_index):
    # vllm config
    llm = LLM(
        MODEL,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=NUM_GPUS,
        max_num_seqs=16,
    )
    
    # prompt the llm
    sampling_param = SamplingParams(temperature=1, max_tokens=4096, top_p=0.95)
    
    for split_name in SPLIT_NAMES:
        # load in the json
        distributed_list = create_jsons(split_name, load_index)
        print(f"Loaded prompts from prompts_{split_name}_{load_index}.json")
        print(f"Total number of JSON files loaded: {len(distributed_list)}")
        
        # Process the specific file
        output = run_inference(llm, sampling_param, distributed_list[0])
        dump_json_output(output, load_index, split_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with configurable LOAD_INDEX")
    parser.add_argument("--load_index", type=int, default=1, help="Index to load prompts from (default: 1)")
    args = parser.parse_args()

    main(args.load_index)