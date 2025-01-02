# Octopi-S
## Environment Setup
1. In a conda environment with PyTorch / CUDA available, run `pip install -r requirements.txt`.
2. Install `uvicorn`.

## Data
### Processing PhysiCLeAR
1. Create the directory `octopi-s/octopi_s/data/tactile_datasets`.
2. Download the [PhysiCLeAR dataset]().
3. Copy the PhysiCLeAR dataset into `octopi-s/octopi_s/data/tactile_datasets/physiclear`.

## Testing
### Model Weights
1. Download Octopi-S [checkpoints]().
2. Run `python octopi_s/process_datasets.py --dataset_path 

### RAG

### Evaluation
1. Run `python octopi_s/utils/evaluate_llm.py --llm_preds_path {preds.json}`.

## Demo
### Running
1. Change directory into `octopi-s/octopi_s`.
2. Run `uvicorn demo:app --host=0.0.0.0 --port=8000 --log-level=debug --reload`.

## Training
### Encoder

### Multimodal LLM