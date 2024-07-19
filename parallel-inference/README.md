When you're inside src, you can use the following command to run the program:
(change: DATASET_NAME = "verifiers-for-code/Combined-Python-450k-filtered")
```bash
#first run
python3 scripts/generate_prompts.py
```
Then, for inference
Set cuda_visible to be the gpu index, and then load_index to be the index 
```bash
#replace x with gpu index, y (load_index) with the .json split index
CUDA_VISIBLE_DEVICES="x" python3 scripts/inference.py --load_index y
```

For example, if your gpu indices are 3, 4, 5, 7 and indices are 0, 1, 2, 3
```bash
CUDA_VISIBLE_DEVICES="3" python3 scripts/inference.py --load_index 0
CUDA_VISIBLE_DEVICES="4" python3 scripts/inference.py --load_index 1
CUDA_VISIBLE_DEVICES="5" python3 scripts/inference.py --load_index 2
CUDA_VISIBLE_DEVICES="7" python3 scripts/inference.py --load_index 3
```
(run this in tmux pls)

Then, stitch the results together (change: COLUMN_NAME = "llama3_70b_instruct" and DATASET_NAME = "verifiers-for-code/Combined-Python-450k-filtered")
```bash
python3 scripts/stitch_results.py
```