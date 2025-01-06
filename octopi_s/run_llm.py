import itertools
import os
import re 
import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import json
import numpy as np
from utils.dataset import *
from utils.llm import *
from utils.encoder import *
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
import random
import yaml
from datetime import datetime
from scipy.stats import kendalltau



def run_llm(configs, exp_id, g, device, peft):
    # Prepare RAG embeddings for scenario reasoning
    reason = len(configs["reasoning_files"]) > 0
    if reason:
        if configs["rag"]:
            print("\nGenerating RAG embeddings...")
            tactile_vificlip, tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
            if configs["rag_generate_embeddings"]:
                generate_rag_embeddings(configs, load_exp_configs, tactile_vificlip, device, configs["rag_sample_dir"], configs["embedding_dir"])
            del tactile_adapter
            del property_classifier
            saved_embeddings, sample_tactile_paths, object_ids = get_rag_embeddings(configs, device)
        else:
            tactile_vificlip = None
            saved_embeddings = None
            sample_tactile_paths = None
            object_ids = None

    # Load tokenizer and LLM weights
    tokenizer_path, model_path, new_tokens, no_split_module_classes = get_model_details(configs["model_type"])
    os.makedirs(configs["offload_dir"], exist_ok=True)
    f = open(configs["gpu_config"])
    gpu_config = json.load(f)
    model = load_mllm(configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=None)
    tokenizer = model.tokenizer

    # Load datasets
    if configs["use_clip"]:
        image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    train = len(configs["train_files"]) > 0
    test = len(configs["test_files"]) > 0
    if train:
        train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
        train_loader = DataLoader(train_dataset, batch_size=configs["per_device_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if test:
        test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"])
        test_loader = DataLoader(test_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    if reason:
        reasoning_dataset = TactileLLMDataset(image_processor, configs["reasoning_files"], split_name="test", tokenizer=tokenizer, frame_size=configs["frame_size"], flip_p=configs["flip_p"], model_type=configs["model_type"], rag=configs["rag"], tactile_vificlip=tactile_vificlip, saved_embeddings=saved_embeddings, sample_tactile_paths=sample_tactile_paths, object_ids=object_ids, device=device, retrieval_object_num=configs["retrieval_object_num"])
        reasoning_loader = DataLoader(reasoning_dataset, batch_size=configs["per_device_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    # Training parameters
    # 1) LLM
    if train:
        llm_params = []
        if not peft:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "vicuna" in configs["model_type"]:
                    if "lora" in name or "embed_tokens" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if param.requires_grad:
                    llm_params.append(param)
        else:
            for name, param in model.llm.named_parameters():
                if param.requires_grad:
                    llm_params.append(param)
        if len(llm_params) > 0:
            optimizer_llm = torch.optim.AdamW(llm_params, lr=configs["llm_lr"])
            num_steps = int(len(train_loader) / configs["llm_gradient_accumulation_steps"])
            scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=num_steps)

    # 2) Encoder
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    for name, param in model.tactile_adapter.named_parameters():
        param.requires_grad = False
    # for name, param in model.plain_tactile_adapter.named_parameters():
    #     param.requires_grad = False
    # for name, param in model.dotted_tactile_adapter.named_parameters():
    #     param.requires_grad = False

    # 3) Projection
    for name, param in model.project.named_parameters():
        param.requires_grad = not configs["freeze_projection"]
    if not configs["freeze_projection"]:
        project_params = model.project.parameters()
        if train and len(configs["train_files"]) > 0:
            optimizer_project = torch.optim.AdamW(project_params, lr=configs["projection_lr"])
            scheduler_project = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_project, T_max=len(train_loader) / configs["llm_gradient_accumulation_steps"])

    # Training
    if train:
        model.train()
        model.encoder.eval()
        model.tactile_adapter.eval()
        # get trainable/non-trainable model parameter stats
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        all_params = sum([np.prod(p.size()) for p in model.parameters()])
        if peft:
            max_train_steps = configs["max_peft_train_steps"]
        else:
            max_train_steps = configs["max_train_steps"]
        if max_train_steps < len(train_loader):
            print(f"\nFinetuning LLM for {max_train_steps} samples and {int(max_train_steps / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        else:
            print(f"\nFinetuning LLM for {len(train_loader)} samples and {int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        print('Trainable params: {} ({:.2f}%)'.format(trainable_params, trainable_params / all_params * 100,))
        # total_train_loss = 0
        for train_sample_step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
            answer_tokens = answer_tokens.to(device)
            outputs, _ = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices)
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")
            loss = outputs.loss / configs["llm_gradient_accumulation_steps"]
            loss.backward()
            if (train_sample_step + 1) % configs["llm_gradient_accumulation_steps"] == 0:
                # optimizer updates
                if not configs["freeze_projection"]:
                    optimizer_project.step()
                    scheduler_project.step()
                    optimizer_project.zero_grad()
                if len(llm_params) > 0:
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
            if (train_sample_step + 1) >= max_train_steps:
                break
        print("Saving tokenizer and models...")
        tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_id}/tokenizer")
        if peft:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights_peft")
        else:
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_id}/llm_weights")
        torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_encoder.pt")
        torch.save(model.tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/tactile_adapter.pt")
        # torch.save(model.plain_tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/plain_tactile_adapter.pt")
        # torch.save(model.dotted_tactile_adapter.state_dict(), f"{configs['exps_path']}/{exp_id}/dotted_tactile_adapter.pt")
        # TODO: Save others
        torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_id}/project.pt")
        print(f"LLM finetuning done!")

    # Testing
    if test:
        print(f"\nTesting LLM on the test set for description / ranking...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        preds = []
        # beam_dict = {}
        with torch.no_grad():
            for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                question, chat, answer_tokens, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict = batch
                answer_tokens = answer_tokens.to(device)
                if configs["ranking_sampling"] and "rank" in question[0].lower():
                    # Get all object permutations
                    question_str = chat[0]["content"][0]
                    task = question_str.split("Object")[0] + "Object"
                    objects = question_str.split("Object")[1:]
                    objects = [i.replace("\n", "") for i in objects]
                    num_objects = len(objects)
                    object_permutations = list(itertools.permutations(objects))
                    selected_object_permutations_hardness = []
                    selected_object_permutations_roughness = []
                    for i in range(configs["ranking_sampling_num"]):
                        object_permutation = random.choice(object_permutations)
                        # Permute tactile_frames and all_datasets to match object_permutation
                        indices = [objects.index(obj) for obj in object_permutation]
                        permuted_tactile_frames = [tactile_frames[i] for i in indices]
                        permuted_all_datasets = [all_datasets[i] for i in indices]
                        new_question = task + "Object".join([obj + "\n\n" for o, obj in enumerate(object_permutation)])
                        new_question = new_question.strip("\n")
                        permuted_question = chat[:-1]
                        permuted_question[0]["content"] = new_question
                        permuted_question[0]["role"] = "user"
                        permuted_question = [model.tokenizer.apply_chat_template(permuted_question, tokenize=False, add_generation_prompt=True)]
                        _, question_embeds = model(question=permuted_question, tactile_frames=permuted_tactile_frames, answer_tokens=answer_tokens, all_datasets=permuted_all_datasets, all_indices=all_indices, question_embeds_only=True)
                        generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
                        generation_i = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                        generation_i = generation_i.strip().split(tokenizer.eos_token)[0].strip()
                        if "decreasing" not in generation_i:
                            continue
                        rankings_i = generation_i.split("decreasing")[1:]
                        rankings_i = [re.sub(r"[^\d.,]", "", r.strip()).split(",") for r in rankings_i]
                        print(rankings_i)
                        if len(rankings_i[0]) == num_objects:
                            selected_object_permutations_hardness.append(rankings_i[0])
                        if len(rankings_i[1]) == num_objects:
                            selected_object_permutations_roughness.append(rankings_i[1])
                    tau_sum_hardness = {}
                    print(selected_object_permutations_hardness, selected_object_permutations_roughness)
                    for i, object_permutation_i in enumerate(selected_object_permutations_hardness):
                        tau_sum_hardness[i] = 0
                        for j, object_permutation_j in enumerate(selected_object_permutations_hardness):
                            if i != j:
                                tau, _ = kendalltau(object_permutation_i, object_permutation_j)
                                kendall_tau_distance = 1 - tau
                                tau_sum_hardness[i] += kendall_tau_distance
                    tau_sum_roughness = {}
                    for i, object_permutation_i in enumerate(selected_object_permutations_roughness):
                        tau_sum_roughness[i] = 0
                        for j, object_permutation_j in enumerate(selected_object_permutations_roughness):
                            if i != j:
                                tau, _ = kendalltau(object_permutation_i, object_permutation_j)
                                kendall_tau_distance = 1 - tau
                                tau_sum_roughness[i] += kendall_tau_distance
                    try:
                        best_hardness_ranking = min(tau_sum_hardness, key=tau_sum_hardness.get)
                        best_roughness_ranking = min(tau_sum_roughness, key=tau_sum_roughness.get)
                    except ValueError:
                        pass
                _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None)
                # # TODO: Estimate sorting uncertainty
                # first_position_pairwise_dict = get_first_position_pairwise(model, question_embeds, configs, all_objects_dict, tokenizer)
                # beam_dict = get_beam_pairwise_and_sequence(model, question_embeds, configs, all_objects_dict, tokenizer)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                answer_tokens = answer_tokens[0].cpu().numpy()
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                if configs["ranking_sampling"] and "rank" in question[0].lower():
                    print(generation)
                    # TODO: Check if ranking exists in generation
                    # TODO: Permute tactile_frames and all_datasets to match object_permutation
                for i in tactile:
                    sample_paths.append(i[0])
                    data = json.load(open(os.path.join("/".join(i[0].split("/")[:-1]), "data.json"), "r"))
                    all_objects.append(data["object_id"])
                preds.append({
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "question": question,
                    "final_true_answer": answer,
                    "final_generation": generation,
                    # "first_position_pairwise": first_position_pairwise_dict,
                    # "beam": beam_dict,
                })
            if peft:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm_peft.json'
            else:
                preds_json_path = f'{configs["exps_path"]}/{exp_id}/preds/llm.json'
            with open(preds_json_path, 'w') as f:
                json.dump(preds, f, indent=4)
                f.close()
        print(f"LLM testing done!")

    # Reasoning
    if reason:
        print(f"\nRunning LLM on the reasoning set...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()
        # mse_loss_fn = torch.nn.MSELoss()
        all_reason = {}
        sample_no = {}
        with torch.no_grad():
            for reasoning_sample_step, batch in enumerate(tqdm.tqdm(reasoning_loader)):
                all_objects, sample_paths = [], []
                # NOTE: hardcoded for batch size of 1
                chat, tactile_frames, tactile, all_datasets, all_indices, all_objects_dict, scenario, scenario_steps, target, num_candidates, rag_outputs = batch
                generated_chat = []
                scenario = f"{scenario[0]}_{target[0]}"
                check_scenario = False
                if scenario_steps.item() == 0:
                    continue
                if configs["scenarios"] is not None:
                    for scenario_to_check in configs["scenarios"]:
                        if scenario_to_check in scenario:
                            check_scenario = True
                            break
                    if not check_scenario:
                        continue
                if scenario not in all_reason.keys():
                    all_reason[scenario] = []
                    sample_no[scenario] = 1
                else:
                    sample_no[scenario] += 1
                if configs["user_stop_idx"] is not None:
                    chat = chat[:int(configs["user_stop_idx"]*2)]
                for c in range(len(chat)-1):
                    # NOTE: Only for batch size = 1
                    chat[c] = {k:v[0] for k,v in chat[c].items()}
                    if c % 2 == 0:
                        # Question
                        # NOTE: Only for batch size = 1
                        generated_chat.append(chat[c])
                    else:
                        # Answer
                        answer_idx = int((c-1)/2)
                        # RAG and change descriptions
                        if answer_idx in configs["generate_idx"]:
                            # Generate
                            question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                            _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                            # Descriptions / rankings
                            generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                            generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                            generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                            chat[c]["generate"] = True
                            chat[c]["true_answer"] = chat[c]["content"]
                            chat[c]["content"] = generation
                            generated_chat.append(chat[c])
                        else:
                            chat[c]["generate"] = False
                            generated_chat.append(chat[c])
                        # NOTE: Only for one touched object in the guessing game
                        if answer_idx == 0 and configs["rag"]: # NOTE: Assume generate_idx=0 is the description
                            chat[c]["content"] += "\nMost similar objects (in order of decreasing similarity):"
                            for obj_name, obj_descriptions in rag_outputs[0].items():
                                if configs["rag_use_descriptions"]:
                                    chat[c]["content"] += f" {obj_name} ({', '.join(sorted([i[0] for i in obj_descriptions]))});"
                                else:
                                    chat[c]["content"] += f" {obj_name};"
                            chat[c]["content"] = chat[c]["content"][:-1] # NOTE: Remove last comma
                final_question = [tokenizer.apply_chat_template(generated_chat, tokenize=False, add_generation_prompt=True)]
                final_true_answer = chat[-1]["content"][0]
                _, question_embeds = model(question=final_question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
                if configs["reasoning_sampling_num"] == 1:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    final_generation = tokenizer.decode(generation_tokens.sequences[0], skip_special_tokens=True).strip()
                    final_generation = final_generation.strip().split(tokenizer.eos_token)[0].strip()
                else:
                    generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=True, temperature=configs["reasoning_temperature"], num_return_sequences=configs["reasoning_sampling_num"], top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
                    option_generations = {}
                    option_counts = {}
                    option_entropies = {}
                    if configs["reasoning_selection_type"] == "best_of_n":
                        entropies = get_sentence_entropy(generation_tokens, token_start_index=0)
                        max_avg_entropy_per_token = max([i["avg_entropy_per_token"] for i in entropies])
                    for seq_idx, seq in enumerate(generation_tokens.sequences):
                        generation = tokenizer.decode(seq, skip_special_tokens=True).strip()
                        generation = generation.strip().split(tokenizer.eos_token)[0].strip()
                        option = generation.replace("*", "").split("Answer: ")[-1][0]
                        if option not in ["A", "B", "C"]:
                            # NOTE: max_new_tokens is probably not high enough so these are not present in the answer or answer is formatted incorrectly (rare)
                            print(generation)
                            continue
                        if option not in option_generations.keys():
                            option_generations[option] = [generation]
                            option_counts[option] = 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option] = [(max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token]
                        else:
                            option_generations[option].append(generation)
                            option_counts[option] += 1
                            if configs["reasoning_selection_type"] == "best_of_n":
                                option_entropies[option].append((max_avg_entropy_per_token - entropies[seq_idx]["avg_entropy_per_token"]) / max_avg_entropy_per_token)
                    if configs["reasoning_selection_type"] == "majority_voting":
                        # Get random generation from best option
                        most_common_option = max(option_counts, key=option_counts.get)
                        final_generation = random.choice(option_generations[most_common_option])
                    elif configs["reasoning_selection_type"] == "best_of_n":
                        # Weigh with average entropy per token
                        best_option = max(option_entropies, key=lambda k: sum(option_entropies[k]))
                        final_generation = option_generations[best_option][option_entropies[best_option].index(max(option_entropies[best_option]))]
                # if final_true_answer.split("Object ")[1][0] == final_generation.split("Object ")[1][0]:
                #     correct = True
                # else:
                #     correct = False
                # if final_true_answer[0] == final_generation[0]:
                #     correct = True
                # else:
                #     correct = False
                # # 1) Get white-box guessing uncertainties
                # guess_uncertainty_stats, object_scores_dict = get_guess_stats(model.tokenizer, generation_tokens, len(tactile))
                # 2) Get linguistic confidence scores
                # top_option_probabilities = get_linguistic_confidence(model, configs, tokenizer, generated_chat, tactile_frames, all_datasets, all_indices)
                # best_generation_answer = best_generation.split("Answer: ")[-1][0]
                # # 3) Get regression errors
                # all_prop_preds = []
                # all_prop_cls_loss = []
                # for i in tactile:
                #     sample_paths.append(i[0])
                #     data = json.load(open(os.path.join("/".join(i[0].split("/")[:-1]), "data.json"), "r"))
                #     all_objects.append(data["object_id"])
                # for i in range(len(tactile)):
                #     data = json.load(open(os.path.join("/".join(tactile[i][0].split("/")[:-1]), "data.json"), "r"))
                #     tactile_video_features, _, _, _ = tactile_vificlip(tactile_frames[i].to(device), None, None)
                #     # if not configs["prompt_learning"]:
                #     #     tactile_video_features = models["tactile_adapter"](tactile_video_features)
                #     tactile_video_features = model.dotted_tactile_adapter(tactile_video_features)
                #     plain_indices = [i for i, x in enumerate(all_datasets[i][0]) if get_dataset_sensor_type(x) == "plain"]
                #     plain_tactile_video_features = model.plain_tactile_adapter(tactile_video_features)
                #     tactile_video_features_clone = tactile_video_features.clone()
                #     tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                #     prop_preds = property_classifier(tactile_video_features_clone)
                #     properties = torch.from_numpy(np.array([[data["properties"]["hardness"], data["properties"]["roughness"]]]))
                #     prop_cls_loss = mse_loss_fn(prop_preds, properties.to(device))
                #     all_prop_preds.append(prop_preds[0].cpu().numpy().tolist())
                #     all_prop_cls_loss.append(prop_cls_loss.item())
                # reverse_vocab = {v: k for k, v in model.tokenizer.vocab.items()}
                # if "llama-3" in reasoning_loader.dataset.model_type:
                #     answer_tokens = encode_text(tokenizer, final_true_answer + reasoning_loader.dataset.eot_token)
                # else:
                #     answer_tokens = encode_text(tokenizer, final_true_answer + reasoning_loader.dataset.eos_token)
                # answer_tokens = answer_tokens.to(device)
                # target_object = int(reverse_vocab[answer_tokens[2]])
                all_reason[scenario].append({
                    "sample_no": sample_no[scenario],
                    "sample_paths": sample_paths,
                    "all_objects": all_objects,
                    "num_candidates": num_candidates.item(),
                    "chat": generated_chat,
                    "generate_idx": configs["generate_idx"],
                    "user_stop_idx": configs["user_stop_idx"],
                    "reasoning_sampling_num": configs["reasoning_sampling_num"],
                    "reasoning_selection_type": configs["reasoning_selection_type"],
                    "final_true_answer": final_true_answer,
                    "final_generation": final_generation,
                    "option_counts": option_counts,
                    "option_entropies": {k: sum(v) for k, v in option_entropies.items()},
                    # "top_option_probabilities": top_option_probabilities,
                    # "entropy": guess_uncertainty_stats["entropy"],
                    # "max_prob": guess_uncertainty_stats["max_prob"].astype(float),
                    # "max_diff": guess_uncertainty_stats["max_diff"].astype(float),
                    # "mse": all_prop_cls_loss,
                    # "average_mse": sum(all_prop_cls_loss) / len(all_prop_cls_loss),
                    # "chosen_mse": all_prop_cls_loss[target_object-1],
                    # "correct": correct,
                    # "all_prop_preds": all_prop_preds
                })
            # NOTE: Prioritize PEFT LLM?
            if os.path.exists(os.path.join(configs["load_exp_path"], "llm_weights_peft")) or os.path.exists(os.path.join(f'{configs["exps_path"]}/{exp_id}/', "llm_weights_peft")):
                peft = True
            else:
                peft = False
            # Save predictions by scenario
            for scenario in all_reason.keys():
                if peft:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}_peft.json'
                else:
                    reason_json_path = f'{configs["exps_path"]}/{exp_id}/reason/{scenario}.json'
                with open(reason_json_path, 'w') as f:
                    json.dump(all_reason[scenario], f, indent=4)
                    f.close()
        print(f"LLM reasoning done!")
    
    # Clean up
    del model
    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_type = f"run"
    config_path = f'configs/{run_type}.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    exp_name = input("\nExperiment name: ")
    if len(exp_name) == 0:
        exp_name = "debug"
    exp_id = "_llm"
    if len(configs["train_files"]) > 0:
        exp_id += "_train_peft"
    if len(configs["test_files"]) > 0:
        exp_id += "_test"
    if len(configs["reasoning_files"]) > 0:
        exp_id += "_reason"
    if len(exp_name) > 0:
        exp_id = exp_id + f"_{exp_name}"

    # Make experiment folder
    now = datetime.now()
    exp_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_id = exp_date + exp_id
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_id}", exist_ok=True)
    if len(configs["train_files"]) > 0:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/preds", exist_ok=True)
    if len(configs["reasoning_files"]) > 0:
        os.makedirs(f"{configs['exps_path']}/{exp_id}/reason", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_id}/{run_type}.yaml", 'w') as file:
        documents = yaml.dump(configs, file, sort_keys=False)
        file.close()

    # Seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])
    device = f'cuda:{configs["cuda"]}' # for non-LLM models

    # Training and/or testing
    if len(configs["train_files"]) > 0:
        run_llm(configs, exp_id, g, device, peft=False)
        configs["load_exp_path"] = f"{configs['exps_path']}/{exp_id}"
        run_llm(configs, exp_id, g, device, peft=True)
        configs["load_exp_path"] = f"{configs['exps_path']}/{exp_id}"
    if len(configs["train_files"]) == 0 and len(configs["reasoning_files"]) > 0:
        run_llm(configs, exp_id, g, device, peft=False)