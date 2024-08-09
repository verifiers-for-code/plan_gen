import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

def get_top_similar_rows(example_problems, dataset_name, model_id, top_k=1000):
    # Load the model
    model = SentenceTransformer(model_id, model_kwargs=dict(add_pooling_layer=False))

    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")

    # Encode the example problems
    query_embeddings = model.encode(example_problems, prompt_name="query")

    # Encode all input rows from the dataset
    input_texts = dataset['input']
    document_embeddings = model.encode(input_texts, batch_size=32, show_progress_bar=True)

    # Convert to torch tensors
    query_embeddings = torch.from_numpy(query_embeddings)
    document_embeddings = torch.from_numpy(document_embeddings)

    # Calculate similarity scores for each example problem
    all_top_indices = set()
    for query_embedding in tqdm(query_embeddings, desc="Processing example problems"):
        scores = query_embedding @ document_embeddings.T
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
        all_top_indices.update(top_indices)

    # Create a new dataset with the top similar rows
    top_similar_rows = dataset.select(list(all_top_indices))

    return top_similar_rows

EXAMPLE_1= "from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: \""" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \""""
EXAMPLE_2= "from typing import List def separate_paren_groups(paren_string: str) -> List[str]: \""" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups('( ) (( )) (( )( ))') ['()', '(())', '(()())'] \""""
EXAMPLE_3= "def string_sequence(n: int) -> str: \""" Return a string containing space-delimited numbers starting from 0 upto n inclusive. >>> string_sequence(0) '0' >>> string_sequence(5) '0 1 2 3 4 5' \""""

# Example usage
example_problems = [
    EXAMPLE_1, EXAMPLE_2, EXAMPLE_3
]
dataset_name = "verifiers-for-code/CodePython-27k-multiple-plangen"
model_id = "Snowflake/snowflake-arctic-embed-m-v1.5"

# Get top similar rows
top_similar_dataset = get_top_similar_rows(example_problems, dataset_name, model_id)

# Push the dataset
dataset_name = "achintasandia/similar_problems"
top_similar_dataset.push_to_hub(dataset_name)
print(f"Dataset has been pushed to the Hugging Face Hub: {dataset_name}")

# Optional: Print some statistics about the dataset
print(f"Number of rows in the new dataset: {len(top_similar_dataset)}")
print(f"Columns in the dataset: {top_similar_dataset.column_names}")
print(f"Sample row from the dataset:")
print(top_similar_dataset[0])