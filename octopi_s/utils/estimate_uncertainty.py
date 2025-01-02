import torch, json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy
import random


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


if __name__ == '__main__':
    llm_reason_path = "/home/user/Documents/octopi-v2/data/exps/2024_12_05_22_14_25_reason_debug/reason/guess_touch_from_objects_physiclear_tennis_ball_peft.json"
    with open(llm_reason_path, "r") as f:
        data = json.load(f)
        f.close()

    plot_performance_scores(data, "entropy")
    plot_performance_scores(data, "max_prob", reverse=True)
    plot_performance_scores(data, "max_diff", reverse=True)