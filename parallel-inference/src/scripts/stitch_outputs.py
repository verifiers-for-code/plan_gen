from datasets import load_dataset
import json
import os

# Paths
DATA_OUTPUT = "data/outputs/"
SPLIT_NAMES = ["train"]
HF_DATASET_NAME = "verifiers-for-code/Combined-Python-450k-filtered"
ISA_B = True

# Magic constant for column name
COLUMN_NAME = "llama3_70b_instruct"

def load_json_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            content = json.load(f)
            data.extend(content)  # Assuming each file contains a list of entries
    return data

def get_file_paths(data_dir, split_name):
    file_groups = {}
    for file_name in os.listdir(data_dir):
        if file_name.startswith(f"output_{split_name}_") and file_name.endswith(".json"):
            base_name = '_'.join(file_name.split('_')[2:-1])  # Extract the base name
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(os.path.join(data_dir, file_name))
    
    # Sort the file paths in each group
    for base_name in file_groups:
        file_groups[base_name].sort()
    
    return file_groups

def create_column(data):
    # Extracting the text from each entry in the data list
    col = [entry for entry in data]
    return col

def update_dataset_with_col(dataset, split_name):
    file_groups = get_file_paths(DATA_OUTPUT, split_name)
    all_data = []
    for group_name, file_paths in file_groups.items():
        print(f"Loading files for {split_name}, group {group_name}: {file_paths}")
        data = load_json_files(file_paths)
        all_data.extend(data)
    
    print(f"Loaded {len(all_data)} entries for {split_name}. Please make sure this is same len as the respective split")
    print("Example [:1]: ", all_data[:1])
    col = create_column(all_data)
    print("Example new column [:1]: ", col[:1])
    
    # Add the new column to the dataset
    dataset[split_name] = dataset[split_name].add_column(COLUMN_NAME, col)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset(HF_DATASET_NAME)

    for split_name in SPLIT_NAMES:
        dataset = update_dataset_with_col(dataset, split_name)
    
    print("Column Names Train: ", dataset["train"].column_names)
    dataset.push_to_hub(HF_DATASET_NAME)