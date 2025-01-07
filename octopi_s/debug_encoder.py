import collections
import os, torch, tqdm, random, yaml
from utils.dataset import *
from utils.llm import *
from utils.encoder import *
from train_encoder import visualize
from utils.physiclear_constants import get_categorical_labels
from transformers import CLIPImageProcessor, AutoTokenizer
from datetime import datetime
from torch.utils.data import DataLoader


target_samples_path = "data/schaeffler_test_set/samples"


def troubleshoot_encoder(configs, load_exp_configs, models, exp_id, device):
    os.makedirs(f"{configs['exps_path']}/{exp_id}/troubleshoot", exist_ok=True)
    save_path = f"{configs['exps_path']}/{exp_id}/troubleshoot"
    retrieval_json = {}
    prop_preds_json = {}
    # print("\nGenerating RAG embeddings...")
    # generate_rag_embeddings(configs, load_exp_configs, models["tactile_vificlip"], device, configs["rag_sample_dir"], configs["embedding_dir"])
    saved_embeddings, sample_tactile_paths, object_ids = get_rag_embeddings(configs, device)
    saved_embeddings = saved_embeddings.to(device)
    retrieved = 0
    retrieved_objects_ids = {}
    sample_cnt = 0
    wrong_sample_object_ids = {}
    cos = nn.CosineSimilarity(dim=1, eps=1e-08)
    for sample in os.listdir(target_samples_path):
        dataset = sample.split("_")[0]
        sample_path = os.path.join(target_samples_path, sample)
        sample_tactile_frames = os.path.join(sample_path, "tactile")
        sample_data = json.load(open(os.path.join(sample_path, "data.json"), "r"))
        sample_object_id = sample_data["object_id"]
        image_transforms = get_image_transforms(load_exp_configs["frame_size"], dataset, split_name="test", flip_p=load_exp_configs["flip_p"])
        sample_tactile_frames, _ = get_frames(sample_tactile_frames, None, image_transforms, frame_size=load_exp_configs["frame_size"], train=False, return_indices=True)
        sample_tactile_frames = torch.unsqueeze(sample_tactile_frames, dim=0)
        sample_tactile_video_features, _, _, _ = models["tactile_vificlip"](sample_tactile_frames.to(device), None, None)
        dotted_tactile_video_features = models["dotted_tactile_adapter"](sample_tactile_video_features)
        plain_indices = [i for i, x in enumerate([dataset]) if get_dataset_sensor_type(x) == "plain"]
        plain_tactile_video_features = models["plain_tactile_adapter"](dotted_tactile_video_features)
        tactile_video_features_clone = dotted_tactile_video_features.clone()
        tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
        prop_preds = models["property_classifier"](tactile_video_features_clone)
        prop_preds = [round(i, 3) for i in prop_preds.detach().cpu().numpy().tolist()[0]]
        gt_hardness = HARDNESS_RANK_REGRESSION[sample_object_id]
        gt_roughness = ROUGHNESS_RANK_REGRESSION[sample_object_id]
        print(prop_preds)
        hardness_diff = abs(gt_hardness - prop_preds[0])
        roughness_diff = abs(gt_roughness - prop_preds[1])
        if sample_object_id not in prop_preds_json:
            prop_preds_json[sample_object_id] = {
                "hardness": [hardness_diff],
                "roughness": [roughness_diff]
            }
        else:
            prop_preds_json[sample_object_id]["hardness"].append(hardness_diff)
            prop_preds_json[sample_object_id]["roughness"].append(roughness_diff)
        similarities = cos(saved_embeddings, sample_tactile_video_features)
        similarities_topk = torch.topk(similarities, k=configs["retrieval_object_num"])
        similar_objects_ids = [object_ids[i] for i in similarities_topk.indices]
        sample_cnt += 1
        if sample_object_id not in retrieved_objects_ids.keys():
            retrieved_objects_ids[sample_object_id] = [0, 1]
        else:
            retrieved_objects_ids[sample_object_id][1] += 1
        if sample_object_id in similar_objects_ids:
            retrieved += 1
            retrieved_objects_ids[sample_object_id][0] += 1
        else:
            if sample_object_id not in wrong_sample_object_ids.keys():
                wrong_sample_object_ids[sample_object_id] = similar_objects_ids
            else:
                wrong_sample_object_ids[sample_object_id].extend(similar_objects_ids)
        if sample_cnt % 50 == 0:
            print(f"Processed {sample_cnt} samples for retrieval.")
    print("\n")
    for k, v in retrieved_objects_ids.items():
        if k in wrong_sample_object_ids.keys():
            retrieval_json[k] = {
                "retrieved": retrieved_objects_ids[k][0] / retrieved_objects_ids[k][1] * 100,
                "similar_objects": collections.Counter(wrong_sample_object_ids[k]).most_common(),
            }
        else:
            retrieval_json[k] = {
                "retrieved": retrieved_objects_ids[k][0] / retrieved_objects_ids[k][1] * 100,
            }
    with open(os.path.join(save_path, "retrieval.json"), 'w') as f:
        json.dump(retrieval_json, f, indent=4)
        f.close()
    with open(os.path.join(save_path, "prop_preds.json"), 'w') as f:
        json.dump(prop_preds_json, f, indent=4)
        f.close()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = "troubleshoot_encoder"
    if len(exp_name) > 0:
        exp_id = exp_id + f"_{exp_name}"

    # Make experiment folder
    now = datetime.now()
    exp_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_id = exp_date + "_" + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}/viz", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_id}/{run_type}.yaml", 'w') as file:
        documents = yaml.dump(configs, file, sort_keys=False)
        file.close()
    device = f'cuda:{configs["cuda"]}'

    tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
    tactile_vificlip.eval()
    dotted_tactile_adapter.eval()
    plain_tactile_adapter.eval()
    property_classifier.eval()
    models = {
        "tactile_vificlip": tactile_vificlip,
        "dotted_tactile_adapter": dotted_tactile_adapter,
        "plain_tactile_adapter": plain_tactile_adapter,
        "property_classifier": property_classifier,
    }

    print("\nGetting encoder regression results...")
    image_processor = CLIPImageProcessor.from_pretrained(load_exp_configs["use_clip"])
    tokenizer = AutoTokenizer.from_pretrained(load_exp_configs["use_clip"])
    test_prop_reg_dataset = TactilePropertyRegressionDataset(image_processor=image_processor, tokenizer=tokenizer, data_path=configs["data_dir"], split_name="test", datasets=configs["datasets"], frame_size=configs["frame_size"])
    test_prop_reg_loader = DataLoader(test_prop_reg_dataset, batch_size=configs["batch_size"], shuffle=False, collate_fn=regression_collate_fn)
    loaders = {
        "property_regression": test_prop_reg_loader
    }
    visualize(configs, loaders, models, split="test", pca=None, device=device, exp_id=exp_id)

    print("\nTroubleshooting encoder...")
    troubleshoot_encoder(configs, load_exp_configs, models, exp_id, device)