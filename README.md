# Octopi-S
## Environment Setup
1. In a conda environment with PyTorch / CUDA available, run `pip install -r requirements.txt`.
2. Install `uvicorn`.
3. For the steps below, ensure you are in the root directory `octopi-s/` unless otherwise stated.


## Data
### Processing PhysiCLeAR
1. Create the directory `octopi_s/data/tactile_datasets`.
2. Download the [PhysiCLeAR dataset](https://drive.google.com/drive/folders/1Gb6n-nGUQxaiuf1hZrgwSvbpl4SWihN4?usp=sharing).
3. Put the PhysiCLeAR dataset in `octopi_s/data/tactile_datasets/` as `octopi_s/data/tactile_datasets/physiclear/`.
4. Run `python octopi_s/process_datasets.py --dataset_path octopi_s/data/tactile_datasets`.

### Generating Question-Answer (QA) Pairs
1. Set these necessary configs in `configs/generate_qa.yaml`:
  * output_data_dir: {QA pair output directory}
  * description_qa_train_num: {Number of training QA pairs to generate for description and/or ranking}
  * description_qa_test_num: {Number of test QA pairs to generate for description and/or ranking}
  * scenario_qa_test_num: {Number of test QA pairs to generate for scenario reasoning}
2. Run `python octopi_s/generate_qa.py`.
3. Enter the scenario QA ID you want when prompted to make the QA file easily identifiable.


## Testing
### Model Weights
1. Download [Octopi-S model weights]().
2. Put the weights in `octopi_s/data/` as `octopi_s/data/weights/`.

### Encoder
1. Set these configs in `configs/run.yaml`:
  * cuda: {GPU device number}
  * load_exp_path: {Directory of pretrained encoder}
  * retrieval_object_num: {Number of similar embeddings to use for RAG}
  * embedding_dir: {Directory where RAG embeddings were saved from the previous step}
  * target_samples_path: {Directory of processed sample tactile videos to get encoder retrieval performance for},
TODO

### Multimodal LLM
1. Set max GPU memory configs in `configs/gpu_config.json`.
2. Set these configs in `configs/run.yaml`:
  * cuda: {GPU device number}
  * load_exp_path: {Directory of pretrained multimodal LLM}
  * data_dir: {Directory of target samples}
  * rag_generate_embeddings: {Whether to generate RAG embeddings or not using the tactile video encoder the LLM was trained with}
  * rag_sample_dir: {Directory of samples to generated RAG embeddings for}
  * embedding_dir: {Output directory for generated RAG embeddings}
3. Run `python octopi_s/run.py`.
4. After you have generated a prediction JSON file (instructions above) for ranking and/or scenario reasoning, run `python octopi_s/evaluate_llm.py --llm_preds_path {results.json}`.


## Demo
1. Set max GPU memory configs in `configs/gpu_config.json`.
2. Set these necessary configs in `configs/demo.yaml`:
  * cuda: {GPU device number}
  * load_exp_path: 
  * train_encoder: True
  * test_encoder: True
3. Change directory into `octopi_s/`.
4. Run `uvicorn demo:app --host=0.0.0.0 --port=8000 --log-level=debug --reload`.


## Training
Make sure you have processed the dataset(s) you want to train with.
### Encoder
1. You need around XX of GPU memory to train the encoder with...
2. Set these configs in `configs/run.yaml`:
  * cuda: {GPU device number}
  * load_exp_path: null
  * train_encoder: True
  * test_encoder: True
3. Run `python octopi_s/run.py`.

### Multimodal LLM
1. Set these configs in `configs/run.yaml`:
  * cuda: {GPU device number}
  * load_exp_path: {Directory of pretrained encoder / MLLM}
  * train_llm: True
  * train_llm_peft: True
2. Run `python octopi_s/run.py`.