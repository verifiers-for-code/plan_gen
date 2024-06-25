import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from evalplus.data import get_human_eval_plus
import json
# ===== CONFIG ===== #
# Magic constants
MODEL = "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = MODEL.split('/')[-1]
DATASET = "verifiers-for-code/humaneval_plan"
OUTPUT_DIR = MODEL_NAME + "-output"
NUM_GPUS = 1
COLUMN_NAME = "generated_phi3_baseline"
__MAGIC_SPLITTER__ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
MAX_TOKENS = 2048
# ================= #

def main():
    response = f"""
    Below is a self-contained Python script that solves the problem: 
    ```python 
    {__MAGIC_SPLITTER__}
    ```
    """

    dataset = load_dataset(DATASET, split="test")
    eplus = get_human_eval_plus()

    llm = LLM(model=MODEL, 
              tensor_parallel_size=NUM_GPUS, 
              enable_prefix_caching=False, 
              gpu_memory_utilization=0.95, 
              max_model_len=2048, 
              trust_remote_code=True,
              max_num_seqs=16)

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0, top_p=0.95, max_tokens=MAX_TOKENS,
    )

    none_prompts = [create_none_prompts(eplus[task]['prompt'], tokenizer, response, __MAGIC_SPLITTER__) for task in eplus.keys()]

    print(none_prompts[0])

    none_prompts_sol = get_vllm_code(llm, none_prompts, sampling_params)

    dataset = update_dataset_with_solutions(dataset, COLUMN_NAME, none_prompts_sol)
    # dataset.push_to_hub(DATASET)

    none_solutions = [extract_clean_code(code) for code in none_prompts_sol]

    for index, solution in enumerate(none_solutions):
        name = f"HumanEval_{index}"
        os.makedirs(os.path.join(f"{OUTPUT_DIR}/none", name), exist_ok=True)
        with open(os.path.join(f"{OUTPUT_DIR}/none", name, '0.py'), 'w', encoding='utf-8') as f:
            f.write(solution)

    convert_to_jsonl(none_solutions, f"{OUTPUT_DIR}/none/solutions.jsonl")

    print(f"Evaluation command (run this): evalplus.evaluate --dataset humaneval --samples {OUTPUT_DIR}/none")

def create_none_prompts(prompt, tokenizer, response, __MAGIC_SPLITTER__):
    prompt = f"Please provide a self-contained Python script that solves the following problem in a markdown code block. Follow the given plan.\n```\n{prompt.strip()}\n```\n"
    x = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False).split(__MAGIC_SPLITTER__)[0]
    return x

def get_vllm_code(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return [x.outputs[0].text for x in outputs]

def update_dataset_with_solutions(dataset, new_column_name, solutions):
    if new_column_name in dataset.column_names: # Remove the column if it already exists
        dataset = dataset.remove_columns(new_column_name)
    return dataset.add_column(new_column_name, solutions)

def extract_clean_code(text):
    index = text.find("```")
    if index != -1:
        text = text[:index]
    lines = text.splitlines()
    for i, line in enumerate(reversed(lines)):
        if "return" in line:
            last_return_index = len(lines) - i - 1
            return '\n'.join(lines[:last_return_index+1])
    return text

def convert_to_jsonl(solutions, output_file):
    with open(output_file, 'w') as f:
        for i, solution in enumerate(solutions):
            json_line = json.dumps({"task_id": f"HumanEval/{i}", "completion": solution})
            f.write(json_line + '\n')

if __name__ == "__main__":
    main()