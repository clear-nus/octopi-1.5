import argparse
import json
import math
import natsort
from utils.llm import get_rankings


def evaluate_ranking(data):
    property_order_results = {
        "no_ranking": 0,
        "invalid_ranking_count": 0
    }
    d_cnt = 0
    for d in data:
        generation = d["final_generation"]
        answer = d["final_true_answer"]
        if "decreasing" not in answer:
            continue
        if "decreasing" not in generation:
            property_order_results["no_ranking"] += 1
            continue
        generation_hardness_order, generation_roughness_order = get_rankings(generation)
        num_hardness_objects = len(generation_hardness_order)
        num_roughness_objects = len(generation_roughness_order)
        answer_hardness_order, answer_roughness_order = get_rankings(answer)
        if natsort.natsorted(generation_hardness_order) != natsort.natsorted(answer_hardness_order) or natsort.natsorted(generation_roughness_order) != natsort.natsorted(answer_roughness_order):
            property_order_results["invalid_ranking_count"] += 1
        else:
            pairwise_count = sum([i for i in range(num_hardness_objects)])
            if num_hardness_objects not in property_order_results.keys():
                property_order_results[num_hardness_objects] = {
                    "pairwise_count": pairwise_count,
                    "hardness_pairwise_correct": 0,
                    "roughness_pairwise_correct": 0,
                    "count": 1,
                    "hardness_correct": 0,
                    "roughness_correct": 0
                }
            else:
                property_order_results[num_hardness_objects]["pairwise_count"] += pairwise_count
                property_order_results[num_hardness_objects]["count"] += 1
            for i in natsort.natsorted(generation_hardness_order):
                for j in natsort.natsorted(generation_hardness_order):
                    if j <= i:
                        continue
                    else:
                        if generation_hardness_order[i] - generation_hardness_order[j] < 0 and answer_hardness_order[i] - answer_hardness_order[j] < 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
                        elif generation_hardness_order[i] - generation_hardness_order[j] > 0 and answer_hardness_order[i] - answer_hardness_order[j] > 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
                        elif generation_hardness_order[i] - generation_hardness_order[j] == 0 and answer_hardness_order[i] - answer_hardness_order[j] == 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
            if generation_hardness_order == answer_hardness_order:
                property_order_results[num_hardness_objects]["hardness_correct"] += 1
            for i in natsort.natsorted(generation_roughness_order):
                for j in natsort.natsorted(generation_roughness_order):
                    if j <= i:
                        continue
                    else:
                        if generation_roughness_order[i] - generation_roughness_order[j] < 0 and answer_roughness_order[i] - answer_roughness_order[j] < 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
                        elif generation_roughness_order[i] - generation_roughness_order[j] > 0 and answer_roughness_order[i] - answer_roughness_order[j] > 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
                        elif generation_roughness_order[i] - generation_roughness_order[j] == 0 and answer_roughness_order[i] - answer_roughness_order[j] == 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
            if generation_roughness_order == answer_roughness_order:
                property_order_results[num_roughness_objects]["roughness_correct"] += 1
        d_cnt += 1
    print(d_cnt)
    accuracy = {i: {} for i in property_order_results.keys() if type(i) == int and i != 1}
    for cnt, result in property_order_results.items():
        if cnt == 1:
            # Only one object
            continue
        elif type(cnt) == str:
            continue
        else:
            accuracy[cnt] = {
                "hardness_pairwise": result["hardness_pairwise_correct"] / result["pairwise_count"],
                "roughness_pairwise": result["roughness_pairwise_correct"] / result["pairwise_count"],
                "hardness": result["hardness_correct"] / result["count"],
                "roughness": result["roughness_correct"] / result["count"]
            }
    return accuracy, property_order_results


def evaluate_reasoning(data):
    correct, cnt = 0, 0
    for d in data:
        generation = d["final_generation"].replace("*", "").split("Answer: ")[-1][0]
        # generation = d["final_generation"].replace("*", "").split("Answer: ")[-1].split(")")[0][-1]
        answer = d["final_true_answer"][0]
        print(f"Answer: {answer}; Generation: {generation}")
        if generation == answer:
            correct += 1
        cnt += 1
    accuracy = correct / cnt
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_preds_path', help='LLM prediction JSON file')
    args = parser.parse_args()
    with open(args.llm_preds_path, "r") as f:
        data = json.load(f)
        f.close()

    if "/reason/" in args.llm_preds_path:
        # Scenario reasoning
        reasoning_accuracy = evaluate_reasoning(data)
        print(f"\nReasoning accuracy: {reasoning_accuracy}")
    else:
        # Rankings
        ranking_accuracy, property_order_results = evaluate_ranking(data)
        print("\n")
        for k, v in ranking_accuracy.items():
            print(f"{k}: {v}")
        print(f"No rank sample output: {property_order_results['no_ranking']}")
        print(f"Invalid sample count: {property_order_results['invalid_ranking_count']}")

    # from get_temp_results import evaluate_rag_reasoning
    # include = "bristles"
    # exclude = []
    # evaluate_rag_reasoning(data, include, exclude)