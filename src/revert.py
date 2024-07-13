from datasets import load_dataset

dataset = load_dataset("verifiers-for-code/test-gran", revision="fd79ca9dcb28e72b3a4b4013b3606679a8215059")
dataset.push_to_hub("verifiers-for-code/test-gran")