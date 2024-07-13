import sys
import os

print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import load_dataset
from vllm import LLM, SamplingParams
from src.utils import iutils
import torch

# Constants
DATASET = "verifiers-for-code/Python-Alpaca-22k-filtered"
MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_UTILIZATION = 0.93
TEMPERATURE = 1.0
MAX_TOKENS = 1024
TOP_P = 0.95
PROMPT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
)
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
SYS_PROMPT = PROMPT_ROOT + "/system_prompts/plangen_27k.txt"
FEW_SHOT_ROOT = PROMPT_ROOT + "/few_shots/plangen_27k/"
EMAIL_FORMAT = PROMPT_ROOT + "/formats/plangen_27k.txt"


def read_text_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


email_format = read_text_file(EMAIL_FORMAT)

messages = [
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

llm = LLM(
    MODEL,
    enable_prefix_caching=True,
    # enforce_eager=True,
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
        input = row["instruction"]
        output = row["output"]
        user_content = email_format.format(
            input, output
        )
        prompt = messages + [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def write_outputs_to_json_file(outputs, filename):
    """
    Writes the list of strings 'outputs' to a JSON file.

    Args:
    outputs (list of str): The list of strings to be written to the file.
    filename (str): The name of the file to write the outputs to.
    """
    try:
        with open(filename, "w") as file:
            json.dump(outputs, file, indent=4)
        print(f"Successfully wrote outputs to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def process_for_dataset_split(
    df, column_name="input"
):
    print("Generating Prompts...")
    prompts = make_prompts_for_data(df)
    print("Running Inference")
    outputs = iutils.run_inference(llm, sampling_params, prompts)
    print(len(outputs))
    write_outputs_to_json_file(outputs, FEW_SHOT_ROOT + "outputs.json")
    df = df.add_column(column_name, outputs)
    return df


dataset = load_dataset(DATASET, split="train")
dataset = process_for_dataset_split(dataset)

# push this to hf hub
try:
    dataset.save_to_disk(DATA_ROOT)
except:
    pass
dataset.push_to_hub(DATASET)