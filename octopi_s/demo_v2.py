import ast
from typing import Union
from fastapi import FastAPI
import torch
import natsort
import cv2 as cv
import yaml, os
import torch, os, yaml
from utils.encoder import *
from utils.llm import *
from utils.dataset import *
from utils.demo_utils import *
from transformers import CLIPImageProcessor, AutoProcessor
from transformers.utils import logging
import numpy as np
import shutil
import os
from datetime import datetime
from qwen_vl_utils import process_vision_info


# API
app = FastAPI()
logging.set_verbosity_error()

# Run settings
run_type = f"demo"
demo_config_path = f'../configs/{run_type}.yaml'
demo_configs = yaml.safe_load(open(demo_config_path))
device = f'cuda:{demo_configs["cuda"]}'
load_exp_path = demo_configs["load_exp_path"]
f = open(demo_configs["gpu_config"])
gpu_config = json.load(f)
embedding_history_path = demo_configs["embedding_history_path"]
chat_path = demo_configs["chat_path"]
dataset = "physiclear" # NOTE: Assume the tactile inputs uses the non-dotted GelSight Mini
app.all_items = None

# RAG
tactile_vificlip, dotted_tactile_adapter, plain_tactile_adapter, property_classifier, load_exp_configs = load_encoder(demo_configs, device)
image_transforms = get_image_transforms(load_exp_configs["frame_size"], dataset, split_name="test", flip_p=0)
if demo_configs["rag"]:
    if demo_configs["rag_generate_embeddings"]:
        print("\nGenerating RAG embeddings...")
        generate_rag_embeddings(demo_configs, load_exp_configs, tactile_vificlip, device, demo_configs["rag_sample_dir"], demo_configs["embedding_dir"])
    del dotted_tactile_adapter
    del plain_tactile_adapter
    del property_classifier
    saved_embeddings, sample_tactile_paths, rag_object_ids = get_rag_embeddings(demo_configs, device)
else:
    tactile_vificlip = None
    saved_embeddings = None
    sample_tactile_paths = None
    rag_object_ids = None

# Load models
load_exp_configs = yaml.safe_load(open(os.path.join(load_exp_path, "run.yaml")))
peft = "peft" in demo_configs["load_exp_path"]
tokenizer_path, model_path, new_tokens, no_split_module_classes = get_model_details(load_exp_configs["model_type"])
load_exp_configs.update(demo_configs)
model = load_mllm(load_exp_configs, tokenizer_path, model_path, new_tokens, no_split_module_classes, peft, device, gpu_config, exp_id=None)
if load_exp_configs["use_clip"]:
    image_processor = CLIPImageProcessor.from_pretrained(load_exp_configs["use_clip"])


def save_chat_history(user_input, generation):
    qa = f"###### USER: {user_input}\n\n###### ASSISTANT: {generation}\n\n"
    if not os.path.exists(chat_path):
        write_type = "w"
    else:
        write_type = "a"
    with open(chat_path, write_type) as f:
        f.write(qa)
        f.close()


@app.post("/describe")
def describe_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=True, rank=False)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    return generation


@app.post("/rank")
def rank_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=False, rank=True)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    response_json = {"response": generation}
    ranks = generation.split("Object parts ranked")[1:]
    characters_to_replace = [model.tokenizer.eos_token, "=", ">"]
    for rank in ranks:
        prop = rank.split(":")[0].split()[-1]
        rank = rank.split(":")[-1]
        for character in characters_to_replace:
            rank = rank.replace(character, "")
        rank = rank.split(",")
        response_json[prop] = [i.strip().strip(".") for i in rank]
    return response_json


@app.post("/describe_and_rank")
def describe_rank_objects(object_ids: str):
    object_ids = [int(i.strip()) for i in object_ids.split(",")]
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, all_embeds, question, tactile_paths_flattened = describe_rank(model, tactile_vificlip, demo_configs, load_exp_configs, object_ids, image_transforms, device, image_processor, new_tokens, saved_embeddings, sample_tactile_paths, rag_object_ids, prev_embeds, describe=True, rank=True)
    torch.save(all_embeds, embedding_history_path)
    save_chat_history(question, generation)
    response_json = {"response": generation}
    ranks = generation.split("Object parts ranked")[1:]
    characters_to_replace = [model.tokenizer.eos_token, "=", ">"]
    for rank in ranks:
        prop = rank.split(":")[0].split()[-1]
        rank = rank.split(":")[-1]
        for character in characters_to_replace:
            rank = rank.replace(character, "")
        rank = rank.split(",")
        response_json[prop] = [i.strip().strip(".") for i in rank]
    return response_json


@app.post("/describe_rgb")
def describe_rgb(prompt: str):
    # NOTE: Does not save into chat history or embedding history, only for demo purposes on the UI
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": demo_configs["image_path"],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.llm.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generation = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = {
        "response": generation,
    }
    objects = generation.split("Object 1")[-1].split("\n")
    final_objects = []
    for obj in objects:
        final_objects.append(obj.split(":")[-1].split(",")[-1].replace(".", "").strip().lower())
    response["objects"] = final_objects
    return response


@app.post("/guess_from_objects")
def guess_touch_given_objects(object_candidates: str):
    object_candidates_options = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }
    object_candidates = [f"{object_candidates_options[i]}) {obj.strip()}" for i, obj in enumerate(object_candidates.split(','))]
    object_candidates_text = ', '.join(object_candidates)
    task_prompt = f"Determine which option the above object is likely to be: {object_candidates_text}?\nFollow the steps below: 1. Select the surface texture descriptions (note: each part of an object contains a different salient texture) that help to distinguish between the given options. 2. Give a succinct case for each option using the selected descriptions. 3. Select the best option and format your answer in the format 'Answer: <letter>) <name> is the most likely option because <reason(s)>'."
    messages = [
        {"role": "user", "content": task_prompt}
    ]
    question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_embeds = process_user_input(question_template, image_processor, model, model.tokenizer, device, new_tokens, load_exp_configs["frame_size"], image_transforms)
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds)
    embeds = torch.cat([question_embeds, generation_embeds], dim=1)
    torch.save(embeds, embedding_history_path)
    save_chat_history(task_prompt, generation)
    return generation


@app.post("/ask")
def ask(query: str):
    messages = [
        {"role": "user", "content": query}
    ]
    question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_embeds = process_user_input(question_template, image_processor, model, model.tokenizer, device, new_tokens, load_exp_configs["frame_size"], image_transforms)
    if os.path.exists(embedding_history_path):
        prev_embeds = torch.load(embedding_history_path, map_location=device, weights_only=True)
    else:
        prev_embeds = None
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds)
    embeds = torch.cat([question_embeds, generation_embeds], dim=1)
    torch.save(embeds, embedding_history_path)
    save_chat_history(query, generation)
    return generation


@app.post("/get_response")
def get_response(query: str):
    prompt = 'You are interfacing with a robot that can sense tactile information and can perform inference on the tactile signals. A user will give a query, and you must identify the most appropriate category that should be sent to the robot. There are 6 categories. Your answer must be in a dictionary format. If there is a "/" in the format specified, it is to separate possible answers.\
        \
        Category 1: "describe and rank"\
        Function: Firstly, ask the robot to describe the tactile properties of the object(s) in the scene. The "objects" will be "1", "2", or "3" if the user asked for either one of these to be described, otherwise, if it is not certain which object, "objects" will be "4". If multiple items are to be described, the objects should be separated by a comma, for example "1,2,3". Secondly, ask the robot to rank the objects by a given criteria, which is either "hardness" or "roughness". If no such criteria was given, the "criteria" will be "uncertain".\
        Format: {"category": 1, "objects": "1"/"2"/"3"/"4", "criteria": "hardness"/"roughness"/"uncertain"}\
        Example query: Describe objects 1 and 2 and rank them by hardness.\
        Example answer: {"category": 1, "objects": "1,2", "criteria": "hardness"}\
        \
        Category 2: "describe"\
        Function: Ask the robot to describe the tactile properties of the object(s) in the scene. The "objects" will be "1", "2", or "3" if the user asked for either one of these to be described, otherwise, if it is not certain which object, "objects" will be "4". If multiple items are to be described, the objects should be separated by a comma, for example "1,2,3".\
        Format: {"category": 1, "objects": "1"/"2"/"3"/"4"}\
        Example query: Describe object 1.\
        Example answer: {"category": 2, "objects": "1"}\
        \
        Category 3: "rank"\
        Function: Ask the robot to rank objects in the scene by a given criteria, which is either "hardness" or "roughness". If no such criteria was given, the "criteria" will be "uncertain". The "objects" will be "1", "2", or "3" if the user asked for either one of these to be described, otherwise, if it is not certain which object, "objects" will be "4". If multiple items are to be described, the objects should be separated by a comma, for example "1,2,3". \
        Format: {"category": 3, "objects": "1"/"2"/"3"/"4", "criteria": "hardness"/"roughness"/"uncertain"}\
        Example query: Please rank objects 1 and 4 by hardness.\
        Example answer: {"category": 3, "objects": "1,4", "criteria": "hardness"}\
        \
        Category 4: "guess from objects"\
        Function: Ask the robot to infer the most likely object given a tactile reading of the object and a list of candidate object names.\ \
        Format: {"category": 4}\
        Example query: Which object is it?\
        Example answer: {"category": 4}\
        \
        Category 5: "prompt"\
        Function: Ask the robot to describe what they see on the table, purely from vision, and not describe them from tactile feedback.\
        Format: {"category": 5}\
        Example query: Which objects do you see?\
        Example answer: {"category": 5}\
        \
        Category 6: "ask"\
        Function: This is a catch-all category for queries that do not fulfil any of the above categories. If the query involves some kind of describe, rank, or guessing, it must never be classified under this category.\ \
        Format: {"category": 6}\
        Example query: The item is not a tennis ball. \
        Example answer: {"category": 6}\
        \
        ------------\
        USER: ' + query.lower() + '\
        \
        ANSWER:'
    messages = [
        {"role": "user", "content": prompt}
    ]
    question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    question_embeds = []
    question_embeds.append(model.llm.get_input_embeddings()(torch.unsqueeze(encode_text(model.tokenizer, question_template), 0).to(device)))
    question_embeds = torch.cat(question_embeds, dim=1)
    generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds=None)
    print("Processed user query:", generation)
    generation = generation.strip().replace("<|im_end|>", "")
    try:
        generation_dict = ast.literal_eval(generation)
    except SyntaxError:
        command = "error"
        response = {
            "response": "The generation is not in a correct syntax. Could you word your query differently and try again?"
        }
    # Process generation
    answer_type = generation_dict["category"]
    if answer_type == 1:  # "describe" in text and "rank" in text:
        command = "describe and rank"
        object_ids = generation_dict["objects"]
        if generation_dict["criteria"] == "hardness" or generation_dict["criteria"] == "roughness":
            response = describe_rank_objects(object_ids)
            try:
                response["target_ranking"] = response[generation_dict["criteria"]]
            except KeyError:
                response = {
                "response": "I could not understand what criteria you want me to sort the items by. Could you try again?"
            }
        else:
            response = {
                "response": "I could not understand what criteria you want me to sort the items by. Could you try again?"
            }
    elif answer_type == 2:  # "describe" in text:
        command = "describe"
        object_ids = generation_dict["objects"]
        response = describe_objects(object_ids)
        response = {
            "response": response
        }
    elif answer_type == 3:  # "rank" in text:
        command = "rank"
        object_ids = generation_dict["objects"]
        if generation_dict["criteria"] == "hardness" or generation_dict["criteria"] == "roughness":
            response = rank_objects(object_ids)
            try:
                response["target_ranking"] = response[generation_dict["criteria"]]
            except KeyError:
                response = {
                "response": "I could not understand what criteria you want me to sort the items by. Could you try again?"
            }
        else:
            response = {
                "response": "I could not understand what criteria you want me to sort the items by. Could you try again?"
            }
    ## handling guess from objects (given the list of objects, what is it?)
    elif answer_type == 4:  # "object" in text:
        command = "guess from objects"
        if app.all_items is None:
            response = "I have not seen any items yet. Please let me see the item(s) first (after taking a RGB picture)."
        else:
            response = guess_touch_given_objects(app.all_items)
            # response = "The last items I've seen with vision are: " + app.all_items + ". " + response
        response = {
            "response": response
        }
    elif answer_type == 5:  # "what is this item: " in text:
        command = "prompt"
        rgb_prompt = "Identify the three central objects, which can either be fruits or balls with non-visual details necessary for tactile reasoning, from right to left. Format your answer as 'Object 1: details, object name.\nObject 2: details, object name.\nObject 3: details, object name.' with less than 5 words each. You must specify the name of the object as the last descriptor for each object.\nExample: Object 1: red, apple.\nObject 2: yellow, banana.\nObject 3: green, tennis ball."
        response = describe_rgb(rgb_prompt)
        app.all_items = ", ".join(response["objects"])
    elif answer_type == 6:
        # treat the rest as an ask command.
        command = "ask"
        response = ask(query)
        response = {
            "response": response
        }
    else:
        response = "Sorry, I could not understand what you are asking for. Could you try again?"
        command = "error"
    
    response["command"] = command
    return response


# @app.post("/get_response")
# def get_response(query: str):
#     prompt = 'You are interfacing with a robot that can sense tactile information and can perform inference on the tactile signals. A user will give a query, and you must identify the most appropriate category that should be sent to the robot. There are 6 categories. Your answer must follow a dictionary format: {CATEGORY_NUMBER: ADDITIONAL DETAILS or NONE if no details are needed}\
#         \
#         Category 1: "describe and rank"\
#         Function: Ask the robot to describe the objects in the scene and rank them by a given criteria, which is either "hardness" or "roughness". If no such criteria was given, the ADDITIONAL DETAILS will be "uncertain".\
#         Example query: Describe objects 1 and 2 and rank them by hardness.\
#         Format: {1: hardness/roughness/uncertain}\
#         \
#         Category 2: "describe"\
#         Function: Ask the robot to describe the tactile properties of the object(s) in the scene. The ADDITIONAL_DETAILS will be 1, 2, or 3 if the user asked for either one of these to be described, otherwise, if it is not certain which object, ADDITIONAL_DETAILS will be 4. If multiple items are to be described, ADDITIONAL_DETAILS should be "1,2,3"\
#         Format: {2: 1/2/3/4}\
#         \
#         Category 3: "rank"\
#         Function: Ask the robot to rank objects in the scene by a given criteria, which is either "hardness" or "roughness". If no such criteria was given, the ADDITIONAL DETAILS will be "uncertain".\
#         Example query: Please rank the items by hardness.\
#         Format: {3: hardness/roughness/uncertain}\
#         \
#         Category 4: "guess from objects"\
#         Function: Ask the robot to infer the most likely object given a tactile reading of the object and a list of candidate object names.\
#         Example query: Which object is it?\
#         Format: {4: None}\
#         \
#         Category 5: "prompt"\
#         Function: Ask the robot to describe what they see on the table, purely from vision, and not describe them from tactile feedback.\
#         Format: {5: None}\
#         \
#         Category 6: "ask"\
#         Function: This is a catch-all category for queries that do not fulfil any of the above categories. If the query involves some kind of describe, rank, or guessing, it must never be classified under this category.\
#         Example query: The item is not a tennis ball. \
#         Format: {6: None}\
#         \
#         ------------\
#         USER: ' + query.lower() + '\
#         \
#         ANSWER:'
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
#     question_template = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     question_embeds = []
#     question_embeds.append(model.llm.get_input_embeddings()(torch.unsqueeze(encode_text(model.tokenizer, question_template), 0).to(device)))
#     question_embeds = torch.cat(question_embeds, dim=1)
#     generation, generation_embeds, question_embeds = generate(question_embeds, model, demo_configs["max_new_tokens"], prev_embeds=None)
#     print(generation)
#     generation = generation.strip().replace("{", "").replace("}", "").replace("<|im_end|>", "")
#     # Process generation
#     answer_type = int(generation[0])
#     context = "".join(generation.split(":")[1:]).strip()
#     rank_criteria = None
#     if answer_type == 1:  # "describe" in text and "rank" in text:
#         command = "describe and rank"
#         context = context.replace("\"", "")
#         if context.lower() == "hardness" or context.lower() == "roughness":
#             rank_criteria = context.replace("\"", "")
#             object_ids = "1,2,3"
#             response = describe_rank_objects(object_ids)
#         else:
#             response = {
#                 "response": "I could not understand what criteria you want me to sort the items by. Could you try again?"
#             }
#     elif answer_type == 2:  # "describe" in text:
#         command = "describe"
#         object_ids = context
#         print(object_ids)
#         response = describe_objects(object_ids)
#         response = {
#             "response": response
#         }
#     elif answer_type == 3:  # "rank" in text:
#         command = "rank"
#         context = context.replace("\"", "")
#         if context.lower() == "hardness" or context.lower() == "roughness":
#             rank_criteria = context
#             object_ids = "1,2,3"
#             response = rank_objects(object_ids)
#         else:
#             response = {
#                 "I could not understand what criteria you want me to sort the items by. Could you try again?"
#             }
#     ## handling guess from objects (given the list of objects, what is it?)
#     elif answer_type == 4:  # "object" in text:
#         command = "guess from objects"
#         response = guess_touch_given_objects(app.all_items)
#         response = {
#             "response": response
#         }
#     elif answer_type == 5:  # "what is this item: " in text:
#         command = "prompt"
#         rgb_prompt = "Identify the three central objects, which can either be fruits or balls with non-visual details necessary for tactile reasoning, from right to left. Format your answer as 'Object 1: details, object name.\nObject 2: details, object name.\nObject 3: details, object name.' with less than 5 words each. You must specify the name of the object as the last descriptor for each object.\nExample: Object 1: red, apple.\nObject 2: yellow, banana.\nObject 3: green, tennis ball."
#         response = describe_rgb(rgb_prompt)
#         app.all_items = ", ".join(response["objects"])
#     elif answer_type == 6:
#         # treat the rest as an ask command.
#         command = "ask"
#         response = ask(query)
#         response = {
#             "response": response
#         }
#     else:
#         response = "Sorry, I could not understand what you are asking for. Could you try again?"
#         command = "error"
    
#     response["command"] = command
#     if rank_criteria is not None:
#         response["target_ranking"] = response[rank_criteria]
#     return response


@app.post("/reset")
def reset_llm_history():
    if os.path.exists(embedding_history_path):
        os.remove(embedding_history_path)
    if os.path.exists(chat_path):
        os.remove(chat_path)
    for sample in os.listdir(demo_configs["demo_path"]):
        sample_path = os.path.join(demo_configs["demo_path"], sample)
        if os.path.isdir(sample_path):
            num_parts = len([i for i in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, i))])
            if num_parts <= 1:
                # One object part only
                if "frames" in os.listdir(sample_path):
                    shutil.rmtree(os.path.join(sample_path, "frames"))
            else:
                # Multiple object parts
                for part in natsort.natsorted(os.listdir(sample_path)):
                    part_path = os.path.join(sample_path, part)
                    if "frames" in os.listdir(part_path):
                        shutil.rmtree(os.path.join(part_path, "frames"))
    # app.all_items = None
    return {"status": "done"}