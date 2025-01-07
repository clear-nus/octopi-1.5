# Octopi-S
## Environment Setup
1. In a conda environment with PyTorch / CUDA available, run `pip install -r requirements.txt`.
2. Install `uvicorn`.
3. For the steps below, ensure you are in the root directory `octopi-s/` unless otherwise stated.


## Data
### Processing PhysiCLeAR
1. Create the directory `octopi_s/data/tactile_datasets`.
2. Download the [PhysiCLeAR dataset](https://drive.google.com/drive/folders/1qwMrXQO0um2TXSN2KZ8trvW09go04VT0?usp=sharing).
3. Put the PhysiCLeAR dataset in `octopi_s/data/tactile_datasets/` as `octopi_s/data/tactile_datasets/physiclear/`.
4. Run `python octopi_s/process_datasets.py --dataset_path octopi_s/data/tactile_datasets`.

### Generating Question-Answer (QA) Pairs
1. Set configs in `configs/generate_qa.yaml`.
2. Run `python octopi_s/generate_qa.py`.
3. Enter the scenario QA ID you want when prompted to make the QA file easily identifiable.


## Testing
### Model Weights
1. Download [Octopi-S model weights](https://drive.google.com/drive/folders/1rD3ZE-nqGKhxStjPWV6sbN22uN1s0JDr?usp=sharing).
2. Put the weights in `octopi_s/data/` as `octopi_s/data/weights/`.

### Multimodal LLM
1. Set max GPU memory configs in `configs/gpu_config.json`.
2. Set configs in `configs/run.yaml` (note: put at least one file path in `test_files` or `reasoning_files` for it to test and set `train_files: []`).
3. Set `load_exp_path: octopi_s/data/weights` to use our model weights.
4. Run `python octopi_s/run_llm.py`.
5. Enter the experiment ID you want when prompted to make the experiment directory easily identifiable.
6. After you have generated a prediction JSON file (instructions above) for ranking and/or scenario reasoning, run `python octopi_s/evaluate_llm.py --llm_preds_path {path/to/results.json}`.


## Demo
1. Change directory into `octopi_s/`.
2. Set max GPU memory configs in `configs/gpu_config.json`.
3. Set configs in `configs/demo.yaml` (note: absolute paths are preferred).
4. Set `load_exp_path: octopi_s/data/weights` to use our model weights.
5. For a `demo_path: ../data/demo` and `image_path: ../data/demo_videos/demo/rgb.png`, structure your directory like:
├── configs
│   └── ...
├── data
│   └── demo
│       │── 1
│       │   └── item.mov
│       │── 2
│       │   │── 1
│       │   │   └── item.mov
│       │   └── 2
│       │       └── item.mov
│       ├── ...
│       └── rgb.png
├── octopi_s
│   └── ...
└── ...
where `../data/demo/1` contains the tactile video of an object with only one unique part (texture-wise) while `../data/demo/2` is an object with two unique parts.
6. Run `uvicorn demo:app --host=0.0.0.0 --port=8000 --log-level=debug --reload`.
7. Refer to the [API documentation](https://github.com/clear-nus/octopi-s/wiki/API) for more information on usage.


## Training

<!-- TODO -->
### Encoder
1. Set configs in `configs/run.yaml`.
    * Set `load_exp_path` if you want to start from a checkpoint.
2. Run `python octopi_s/train_encoder.py` (note: ou need around XX of GPU memory to train the encoder with...).
3. Enter the experiment ID you want when prompted to make the experiment directory easily identifiable.

### Multimodal LLM
1. Set configs in `configs/run.yaml` (note: you must put at least one file path in `train_files` for it to train and set `test_files` and / or `reasoning_files` if you want it to test / reason as well).
2. Run `python octopi_s/run_llm.py`.
3. Enter the experiment ID you want when prompted to make the experiment directory easily identifiable.