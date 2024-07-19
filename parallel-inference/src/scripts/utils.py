import os

def read_text_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "r") as f:
        content = f.read()
    return content

def run_inference(llm, sampling_params, prompts):
    outputs = llm.generate(
        prompts,
        sampling_params,
    )
    outputs = [x.outputs[0].text for x in outputs]
    return outputs