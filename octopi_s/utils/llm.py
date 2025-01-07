import torch
from torch import nn
from .encoder import *
from .dataset import get_frames, encode_text, get_dataset_sensor_type
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from accelerate import infer_auto_device_map, init_empty_weights


def get_model_details(model_type):
    if model_type == "llama-3.1-8b":
        tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        new_tokens = ["<|start_tactile_id|>", "<|end_tactile_id|>"]
        no_split_module_classes = ["LLaMADecoderLayer", "LlamaDecoderLayer"]
    elif model_type == "qwen2.5-7b":
        tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        new_tokens = ["<|tactile_start|>", "<|tactile_end|>"]
        no_split_module_classes = ["Qwen2DecoderLayer", "Qwen2MLP"]
    elif model_type == "qwen2-vl-7b":
        tokenizer_path = "Qwen/Qwen2-VL-7B-Instruct"
        model_path = "Qwen/Qwen2-VL-7B-Instruct"
        new_tokens = ["<|tactile_start|>", "<|tactile_end|>"]
        no_split_module_classes = ["Qwen2VLDecoderLayer", "Qwen2MLP"]
    return tokenizer_path, model_path, new_tokens, no_split_module_classes


def add_new_tokens(llm, tokenizer, new_tokens):
    new_tokens = list(set(new_tokens) - set(tokenizer.vocab.keys()))
    n_new_tokens = tokenizer.add_tokens(new_tokens)
    print(f"{n_new_tokens} tokens added to tokenizer.")
    llm.resize_token_embeddings(len(tokenizer))
    if n_new_tokens > 0:
        with torch.no_grad():
            input_embeddings_avg = llm.model.embed_tokens.weight[:-n_new_tokens].mean(axis=0, keepdim=True)
            llm.model.embed_tokens.weight[-n_new_tokens:] = input_embeddings_avg


def load_mllm(configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=None):
    if configs["gpu_config"] is not None:
        if configs["load_exp_path"] is not None:
            if os.path.exists(os.path.join(configs["load_exp_path"], "tokenizer")):
                tokenizer_path = os.path.join(configs["load_exp_path"], "tokenizer")
                print("Loading trained tokenizer!")
            if os.path.exists(os.path.join(configs["load_exp_path"], "llm_weights")):
                model_path = os.path.join(configs["load_exp_path"], "llm_weights")
                print("Loading trained LLM!")
        if configs["model_type"] == "qwen2-vl-7b":
            gpu_max_mem_config = {}
            for k, v in gpu_config.items():
                gpu_max_mem_config[int(k)] = v
            with init_empty_weights():
                llm = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", offload_folder=configs["offload_dir"])
            device_map = infer_auto_device_map(
                llm, max_memory = gpu_max_mem_config, no_split_module_classes=no_split_module_classes
            )
            del llm
            llm = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"])
            for name, module in llm.named_modules():
                print(f"{name}: {module.__class__.__name__}")
        else:
            with init_empty_weights():
                config = AutoConfig.from_pretrained(model_path)
                auto_model = AutoModelForCausalLM.from_config(config)
            gpu_max_mem_config = {}
            for k, v in gpu_config.items():
                gpu_max_mem_config[int(k)] = v
            device_map = infer_auto_device_map(
                auto_model, max_memory = gpu_max_mem_config, no_split_module_classes=no_split_module_classes
            )
            llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True, padding_side="left")
        print("Loaded LLM and tokenizer!")
        add_new_tokens(llm, tokenizer, new_tokens)
        print(f"Tokenizer BOS: {tokenizer.bos_token}, EOS: {tokenizer.eos_token}, Pad: {tokenizer.pad_token}")
    # Multimodal LLM instantiation
    if configs["load_exp_path"] is not None:
        prompt_learning = "prompt_learning.yaml" in os.listdir(configs["load_exp_path"])
    model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["dim_context_vision"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm, device=device, new_tokens=new_tokens, prompt_learning=prompt_learning)
    model.to(device)
    # LoRA
    if peft:
        if configs["load_exp_path"] is not None:
            if os.path.exists(os.path.join(configs["load_exp_path"], "llm_weights_peft")):
                llm = PeftModel.from_pretrained(model=llm, model_id=os.path.join(configs["load_exp_path"], "llm_weights_peft"), is_trainable=False, device_map="auto", max_memory=gpu_max_mem_config)
                print("Loaded trained PEFT LLM!")
        else:
            peft_config = LoraConfig(
                r=configs["r"],
                lora_alpha=configs["lora_alpha"],
                lora_dropout=configs["lora_dropout"],
                target_modules=configs["target_modules"],
                bias=configs["bias"],
                inference_mode=False,
                task_type="CAUSAL_LM",
                modules_to_save=configs["modules_to_save"]
            )
            llm_weights_path = f"{configs['exps_path']}/{exp_id}/llm_weights_peft"
            if not os.path.exists(llm_weights_path):
                os.makedirs(llm_weights_path)
                llm_peft = get_peft_model(llm, peft_config)
                llm_peft.save_pretrained(llm_weights_path)
                llm_peft = None
                llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, is_trainable=True, device_map="auto", max_memory=gpu_max_mem_config)
            print("Saved and loaded newly initialized PEFT LLM!")
    model.llm = llm
    print("Loaded LLM!")
    tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(configs, device)
    model.tactile_vificlip = tactile_vificlip
    model.load_exp_configs = load_exp_configs
    del dotted_tactile_adapter
    del plain_tactile_adapter
    del property_classifier
    model.encoder.model.vision_model = tactile_vificlip.clip_model.vision_model
    if os.path.exists(os.path.join(configs["load_exp_path"], "project.pt")):
        model.project.load_state_dict(torch.load(os.path.join(configs["load_exp_path"], "project.pt"), map_location=device, weights_only=True))
        print("Loaded projection module!")
    return model


class MultimodalLLMForCausalLM(nn.Module):
    def __init__(self, tokenizer, clip_model, encoder_output_size, cutoff_len, llm, device, new_tokens, prompt_learning=True):
        super(MultimodalLLMForCausalLM, self).__init__()
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.device = device
        try:
            self.llm_embedding_size = llm.model.embed_tokens.weight.shape[1]
        except AttributeError:
            self.llm_embedding_size = llm.embed_tokens.weight.shape[1]
        self.encoder = CLIPVisionEncoder(clip_model=clip_model)
        self.tactile_adapter = Adapter(input_size=encoder_output_size, output_size=encoder_output_size, residual_ratio=0.5)
        self.project = nn.Sequential(
            nn.Linear(encoder_output_size, self.llm_embedding_size),
            nn.GELU(),
            nn.Linear(self.llm_embedding_size, self.llm_embedding_size),
        )
        self.new_tokens = new_tokens
        self.prompt_learning = prompt_learning

    def get_dummy_token(self, answer_embeds, question_embeds_len):
        batch_size = answer_embeds.shape[0]
        answer_embeds_len = answer_embeds.shape[1]
        index_shift = 0
        # labels are shifted by -1 inside the LlamaForCausalLM source code so tokens < n predict n
        pre_label_token = torch.full((batch_size, question_embeds_len + index_shift), fill_value=-100, dtype=torch.int64, device=self.device)
        post_label_token = torch.full((batch_size, self.cutoff_len - (question_embeds_len + answer_embeds_len + index_shift)), fill_value=-100, dtype=torch.int64, device=self.device)
        return pre_label_token, post_label_token

    def forward(self, question, tactile_frames, answer_tokens, all_datasets, all_indices, question_embeds_only=False):
        # 1) Question embeds
        question_embeds = []
        question = question[0]
        num_tactile = question.count("<tact_tokens>")
        question = question.split("<tact_tokens>")
        for tactile_idx, chunk in enumerate(question):
            # Question
            chunk_embeds = self.llm.get_input_embeddings()(encode_text(self.tokenizer, chunk).to(self.device))
            chunk_embeds = torch.unsqueeze(chunk_embeds, dim=0)
            question_embeds.append(chunk_embeds)
            # Tactile
            if tactile_idx < num_tactile:
                tactile_embeds = self.encoder(tactile_frames[tactile_idx].to(self.device))
                tact_start_embeds = torch.unsqueeze(self.llm.get_input_embeddings()(encode_text(self.tokenizer, self.new_tokens[0]).to(self.device)), dim=0)
                tactile_embeds = self.project(tactile_embeds)
                tact_end_embeds = torch.unsqueeze(self.llm.get_input_embeddings()(encode_text(self.tokenizer, self.new_tokens[1]).to(self.device)), dim=0)
                tactile_embeds = torch.cat([tact_start_embeds, tactile_embeds, tact_end_embeds], dim=1)
                question_embeds.append(tactile_embeds)
        question_embeds = torch.cat(question_embeds, dim=1)
        if question_embeds_only:
            return None, question_embeds
        # 2) answer embeds
        answer_embeds = self.llm.get_input_embeddings()(answer_tokens)
        full_embeds_len = question_embeds.shape[1] + answer_embeds.shape[1]
        question_embeds_len = question_embeds.shape[1]
        batch_size = question_embeds.shape[0]
        # NOTE: padding token embedding index is 0
        padding_embeds = self.llm.get_input_embeddings()(torch.zeros(batch_size, self.cutoff_len - full_embeds_len, device=self.device, dtype=torch.int64))
        # 3) combine embeds
        input_embeds = torch.cat((question_embeds, answer_embeds, padding_embeds), dim=1)
        pre_label_dummy_token, post_label_dummy_token = self.get_dummy_token(answer_embeds, question_embeds_len)
        labels = torch.cat((pre_label_dummy_token, answer_tokens, post_label_dummy_token), dim=1)
        batch_size = answer_embeds.shape[0]
        attention_mask = torch.cat((torch.ones([batch_size, full_embeds_len]), torch.zeros([batch_size, padding_embeds.shape[1]])), dim=1).to(self.device)
        out = self.llm(inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask) #, attention_mask=attention_mask) # pass in embeddings directly: https://huggingface.co/docs/transformers/main/en/model_doc/llama
        return out, question_embeds
    

def process_user_input(user_input, image_processor, model, tokenizer, device, new_tokens, frame_size, image_transforms):
    tact_start, tact_end = new_tokens
    inputs = [""]
    for char in user_input:
        if char == "[" or char == "{":
            inputs.append(char)
        elif char == "]" or char == "}":
            inputs[-1] += char
            inputs.append("")
        else:
            inputs[-1] += char
    question_embeds = []
    for chunk in inputs:
        if "[" not in chunk and "{" not in chunk:
            question_embeds.append(model.llm.get_input_embeddings()(torch.unsqueeze(encode_text(tokenizer, chunk), 0).to(device)))
        else:
            question_embeds.append(model.llm.get_input_embeddings()(torch.unsqueeze(encode_text(tokenizer, tact_start), 0).to(device)))
            frames, indices = get_frames(chunk[1:-1], image_processor, image_transforms, frame_size=frame_size, train=False, return_indices=True)
            tactile_tensors = torch.unsqueeze(frames, dim=0).to(device) # (1, l, c, h, w)
            chunk_embeds = model.project(model.encoder(tactile_tensors))
            question_embeds.append(chunk_embeds)
            question_embeds.append(model.llm.get_input_embeddings()(torch.unsqueeze(encode_text(tokenizer, tact_end), 0).to(device)))
    question_embeds = torch.cat(question_embeds, dim=1)
    return question_embeds







def get_linguistic_confidence(model, configs, tokenizer, generated_chat, tactile_frames, all_datasets, all_indices):
    # Prompting strategy: Top-K
    prompt = "Provide your K best guesses and the probability that each is correct (0% to 100%) in your answer for the following question. Give ONLY the letter of your guesses and probabilities, no other words or explanation. For example:\nG1: <ONLY the letter of first most likely guess; not a complete sentence, just the guess!>, P1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nGk: <ONLY the letter of k-th most likely guess> Pk: <ONLY the probability that Gk is correct, without any extra commentary whatsoever; just the probability!>"
    # Sampling strategy: Self-random
    generated_chat_copy = copy.deepcopy(generated_chat)
    generated_chat_copy[-1]["content"] = f"{prompt}\n\n{generated_chat_copy[-1]['content']}" # NOTE
    final_question = [tokenizer.apply_chat_template(generated_chat_copy, tokenize=False, add_generation_prompt=True)]
    _, question_embeds = model(question=final_question, tactile_frames=tactile_frames, answer_tokens=None, all_datasets=all_datasets, all_indices=all_indices, question_embeds_only=True)
    sample_generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=True, temperature=configs["temperature"], num_return_sequences=configs["sampling_num"], top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
    # Aggregation strategy: Avg-Conf
    top_option_probabilities = {}
    # top_option_generations = {}
    # top_option_count = {}
    for seq in sample_generation_tokens.sequences:
        generation = tokenizer.decode(seq, skip_special_tokens=True).strip()
        generation = generation.strip().split(tokenizer.eos_token)[0].strip()
        answer = generation.split("Answer:\n")[-1]
        top_option = answer.split("G1: ")[-1][0]
        if top_option not in ["A", "B", "C"]:
            # NOTE: max_new_tokens is probably not high enough so these are not present in the answer
            continue
        top_option_probability = answer.split("P1: ")[-1].split("\n")[0]
        # print("\nNEXT >>>")
        # print(generation)
        # print("Top option:", top_option)
        # print("Top option probability:", top_option_probability)
        if top_option not in top_option_probabilities.keys():
            top_option_probabilities[top_option] = [top_option_probability]
            # top_option_generations[top_option] = [generation]
            # top_option_count[top_option] = 1
        else:
            top_option_probabilities[top_option].append(top_option_probability)
            # top_option_generations[top_option].append(generation)
            # top_option_count[top_option] += 1
    # most_common_option = max(top_option_count, key=top_option_count.get)
    # max_confidence_for_most_common_option = max(top_option_probabilities[most_common_option])
    # indices = [i for i, prob in enumerate(top_option_probabilities[most_common_option]) if prob == max_confidence_for_most_common_option]
    # indice = random.choice(indices)
    # best_generation = top_option_generations[most_common_option][indice]
    # print(best_generation)
    return top_option_probabilities


def get_guess_scores(tokens, token_start_index=0, sequence_index=0):
    # NOTE: Does not account for object parts
    guess_scores = tokens.scores[token_start_index][sequence_index]
    return guess_scores


def get_guess_stats(tokenizer, tokens, num_objects):
    guess_uncertainty_stats = {}
    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    guess_scores = get_guess_scores(tokens)
    object_scores = torch.topk(guess_scores, k=num_objects, dim=-1)
    object_scores_dict = {}
    for i in range(len(object_scores.indices)):
        object_idx = object_scores.indices[i]
        object_scores_dict[reverse_vocab[object_idx.item()]] = object_scores.values[i].item()
    # Compute softmax
    guess_scores_np = guess_scores.cpu().numpy()
    exp_logits = np.exp(guess_scores_np - np.max(guess_scores_np))
    softmax_probabilities = exp_logits / np.sum(exp_logits)
    max_prob = np.max(softmax_probabilities)
    reverse_sorted_softmax_probabilities = np.sort(softmax_probabilities)[::-1]
    max_diff = reverse_sorted_softmax_probabilities[0] - reverse_sorted_softmax_probabilities[1]
    entropy = 0
    for i in softmax_probabilities:
        entropy += i * np.log(i + 1e-9)
    entropy = -entropy
    guess_uncertainty_stats = {
        "entropy": entropy,
        "max_prob": max_prob,
        "max_diff": max_diff,
    }
    return guess_uncertainty_stats, object_scores_dict


def get_sentence_entropy(generation_tokens, token_start_index):
    entropies = []
    for seq_idx, seq_tokens in enumerate(generation_tokens.sequences):
        token_probs = []
        for step, logits in enumerate(generation_tokens.scores):
            # Get the token ID at this step for the current sequence
            token_id = seq_tokens[step]
            # Extract logits for the current sequence and compute probabilities
            probs = torch.softmax(logits[seq_idx], dim=-1)
            token_probs.append(probs[token_id].item())
        # Calculate total entropy for the sequence
        total_entropy = -sum(torch.log2(torch.tensor(token_probs)))
        # Normalize by token count (average entropy per token)
        avg_entropy = total_entropy / len(seq_tokens)
        entropies.append({
            "total_entropy": total_entropy.item(),
            "avg_entropy_per_token": avg_entropy.item(),
        })
    return entropies


def plot_performance_scores(data, stat, reverse=False):
    bins = 10
    all_stats = []
    for d in data:
        # d_stat = d["guess_uncertainty_stats"][stat]
        d_stat = d[stat]
        all_stats.append(d_stat)
    min_stat = min(all_stats)
    interval = (max(all_stats) - min_stat) / bins
    performance = [0 for i in range(bins)]
    counts = [0 for i in range(bins)]
    for d in data:
        # d_stat = d["guess_uncertainty_stats"][stat]
        d_stat = d[stat]
        quotient =  (d_stat - min_stat) // interval
        remainder = (d_stat - min_stat) % interval
        bin = quotient
        if bin > 0 and remainder < 1e-10:
            bin -= 1
        bin = int(bin)
        if d["correct"]:
            performance[bin] += 1
        counts[bin] += 1
    if reverse:
        performance = performance[::-1]
        counts = counts[::-1]
    performance = np.cumsum(performance)
    counts = np.cumsum(counts)
    performance = [performance[i] / counts[i] for i in range(len(performance))]
    if reverse:
        plt.plot([min_stat + (i * interval) for i in range(bins-1,-1,-1)], performance)
        plt.gca().invert_xaxis()
    else:
        plt.plot([min_stat + (i * interval) for i in range(bins)], performance)
    plt.axhline(y=np.round(1/d["num_candidates"], 2), color='r', linestyle='dashed')
    plt.xlabel(f"{stat} Threshold")
    plt.ylabel("Cumulative Accuracy")
    plt.show()
    # plt.savefig(f"{configs['exps_path']}/{exp_id}/reason/confusion_matrix_hardness.png")
    plt.clf()


def parse_object_part_rankings(tokens):
    colon_indices = [i for i, x in enumerate(tokens) if x == 25]
    hardness_start_index = colon_indices[-2] + 2
    hardness_end_index = [i for i, x in enumerate(tokens[hardness_start_index:]) if x == "\\"][0] - 2
    roughness_start_index = colon_indices[-1] + 2
    roughness_end_index = [i for i, x in enumerate(tokens[roughness_start_index:]) if x == "."][-1] - 1
    hardness_obj_idx_pose = {}
    roughness_obj_idx_pose = {}
    return hardness_start_index, roughness_start_index


def get_first_position_pairwise(model, input_embeds, configs, all_objects_dict, tokenizer):
    score_dict = {}
    generation_tokens = model.llm.generate(inputs_embeds=input_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
    start_time = datetime.now()
    scores = generation_tokens.scores
    tokens = []
    for token in generation_tokens.sequences[0]:
        tokens.append(token.item())
    if 21006 not in tokens: # "ranked" is not in output
        return {}
    # print(tokens)
    hardness_start_index, roughness_start_index = parse_object_part_rankings(tokens)
    # Get embeddings up till start for hardness
    generation_embeds = model.llm.get_input_embeddings()(generation_tokens.sequences)
    hardness_input_embeddings = torch.cat([input_embeds, generation_embeds[:,:hardness_start_index]], dim=1) # [1, seq_len, embed_size]
    hardness_generation_tokens = model.llm.generate(inputs_embeds=hardness_input_embeddings, max_new_tokens=30, num_beams=1, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
    print("Pairwise hardness:", hardness_generation_tokens.sequences[0])
    hardness_generation = tokenizer.decode(hardness_generation_tokens.sequences[0], skip_special_tokens=True).strip()
    print(hardness_generation)
    # TODO: Get scores for each pair by setting first object
    for k, v in all_objects_dict.items():
        if type(v) == str:
            pass
        elif type(v) == list:
            pass
    hardness_scores = scores[hardness_start_index]
    # print(torch.topk(hardness_scores, k=10, dim=-1))
    # Get embeddings up till start for roughness
    roughness_input_embeddings = torch.cat([input_embeds, generation_embeds[:,:roughness_start_index]], dim=1) # [1, seq_len, embed_size]
    # TODO: Get scores for each pair by setting first object
    end_time = datetime.now()
    print('First position pairwise duration: {}'.format(end_time - start_time))
    return score_dict


def get_beam_pairwise_and_sequence(model, input_embeds, configs, all_objects_dict, tokenizer):
    score_dict = {}
    num_beams = 5
    generation_tokens = model.llm.generate(inputs_embeds=input_embeds, max_new_tokens=configs["max_new_tokens"], num_beams=num_beams, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
    start_time = datetime.now()
    scores = generation_tokens.scores # [seq_len, num_beams, vocab_size]
    for n in range(num_beams):
        score_dict[f"Beam {n+1}"] = {}
        sequences_n = generation_tokens.sequences[n] # [num_beams, seq_len]
        scores_n = [i[n] for i in scores] # [seq_len, num_beams, vocab_size]
        tokens_n = []
        for token in sequences_n: # [num_beams, seq_len]
            tokens_n.append(token.item())
        if 21006 not in tokens_n: # "ranked" is not in output
            skip_count += 1
            continue
        hardness_start_index, roughness_start_index = parse_object_part_rankings(tokens_n)
        hardness_scores = scores[hardness_start_index]
        # print(torch.topk(hardness_scores, k=10, dim=-1))
    end_time = datetime.now()
    print('Beam duration: {}'.format(end_time - start_time))
    return score_dict