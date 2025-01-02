import os, yaml, json
from torchvision import transforms
from utils.encoder import *
from utils.dataset import get_frames, get_dataset_sensor_type


def save_embeddings(configs, device, load_exp_path, sample_dir, embedding_dir):
    # Data transformation
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize(configs["frame_size"], interpolation=3),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
        transforms.CenterCrop(configs["frame_size"])
    ]
    image_transforms = transforms.Compose(transforms_list)
    # Load models
    if "prompt_learning.yaml" in os.listdir(load_exp_path):
        prompt_learning_configs = yaml.safe_load(open(os.path.join(load_exp_path, "prompt_learning.yaml")))
        clip = PromptLearningCLIPModel.from_pretrained(prompt_learning_configs["use_clip"], prompt_learning_configs).to(device)
    else:
        clip = CLIPModel.from_pretrained(configs["use_clip"]).to(device)
    tactile_vificlip = ViFiCLIP(clip, freeze_text_encoder=True, use_positional_embeds=True).to(device)
    if os.path.exists(os.path.join(load_exp_path, "tactile_vificlip.pt")):
        tactile_vificlip.load_state_dict(torch.load(os.path.join(load_exp_path, "tactile_vificlip.pt"), weights_only=True), strict=False)
        print("Loaded tactile ViFi-CLIP!")
    tactile_vificlip.eval()
    # if configs["use_projection_module"]:
    #     plain_tactile_adapter = CLIPRFC(input_size=1024, output_size=1024, residual_ratio=0.5).to(device)
    #     if os.path.exists(os.path.join(load_exp_path, "project.pt")):
    #         plain_tactile_adapter.load_state_dict(torch.load(os.path.join(load_exp_path, "plain_tactile_adapter.pt"), weights_only=True), strict=False)
    #         print("Loaded plain tactile adapter!")
    #     plain_tactile_adapter.eval()
    #     project = nn.Sequential(
    #         nn.Linear(1024, 3584),
    #         nn.GELU(),
    #         nn.Linear(3584, 3584),
    #     ).to(device)
    #     if os.path.exists(os.path.join(load_exp_path, "project.pt")):
    #         project.load_state_dict(torch.load(os.path.join(load_exp_path, "project.pt"), weights_only=True), strict=False)
    #         print("Loaded projection module!")
    #     project.eval()

    # Get sample count
    sample_count = 0
    for sample in os.listdir(sample_dir):
        dataset = sample.split("_")[0]
        if "physiclear" not in dataset and "schaeffler" not in dataset:
            continue
        else:
            sample_path = os.path.join(sample_dir, sample)
            data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
            if data["split"] != "train":
                continue
            sample_count += 1

    saved_count = 0
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    for sample in os.listdir(sample_dir):
        # Save embeddings only for relevant training samples
        dataset = sample.split("_")[0]
        if "physiclear" not in dataset and "schaeffler" not in dataset:
            continue
        sample_path = os.path.join(sample_dir, sample)
        data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
        if data["split"] != "train":
            continue
        embedding_path = os.path.join(embedding_dir, f"{sample}.pt")
        tactile = os.path.join(sample_path, "tactile")
        tactile_frames, _ = get_frames(tactile, None, image_transforms, frame_size=configs["frame_size"], train=False, return_indices=True)
        tactile_frames = torch.unsqueeze(tactile_frames, dim=0)
        tactile_video_features, _, _, _ = tactile_vificlip(tactile_frames.to(device), None, None)
        # if configs["use_projection_module"]:
        #     # NOTE: Assume plain GelSight sensor
        #     tactile_video_features = project(plain_tactile_adapter(tactile_video_features))
        torch.save(torch.squeeze(tactile_video_features.cpu(), dim=0), embedding_path)
        saved_count += 1
        if saved_count % 100 == 0:
            print(f"Saved {saved_count} / {sample_count} embeddings.")
    print("Done!")


if __name__ == "__main__":
    run_type = f"generate_embeddings"
    config_path = f'configs/{run_type}.yaml'
    # Configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    device = f'cuda:{configs["cuda"]}' # for non-LLM models
    load_exp_path = configs["load_exp_path"]
    train_config_path = f'{load_exp_path}/train.yaml'
    # Train configs
    with open(train_config_path, 'r') as file:
        train_configs = yaml.safe_load(file)
    configs["frame_size"] = train_configs["frame_size"]

    save_embeddings(configs, device, load_exp_path, configs["sample_dir"], configs["embedding_dir"])