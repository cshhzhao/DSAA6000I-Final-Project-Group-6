# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, )
import numpy as np
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        default='/data2/Llama-2-7b-hf',
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        default='/data2/Llama-2-7b-hf',
        required=False,
    )
    parser.add_argument(
        "--original_llama_saved_path",
        type=str,
        help="Path to pretrained model",
        default='/data2/Llama-2-7b-hf',
        required=False,
    )    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)

    args = parser.parse_args()

    return args

# 我们要查看假新闻的召回率
def fake_news_recall_computation(all_samples):
    ground_truth = [sample['label_bool'] for sample in all_samples]
    prediction = [sample['pred_bool'] for sample in all_samples]

    # 计算真阳性(TP), 假阳性(FP), 真阴性(TN) 和 假阴性(FN)
    TP = sum(p == 'False' and gt == 'False' for p, gt in zip(prediction, ground_truth))
    FN = sum(p == 'True' and gt == 'False' for p, gt in zip(prediction, ground_truth))

    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return recall
    
def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask = inputs.attention_mask,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)
    # real_output_ids = [
    #         output_id[len(inputs.input_ids[i]) :] for i, output_id in enumerate(generate_ids)
    #     ]
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    # result = tokenizer.batch_decode(real_output_ids,
    #                                 skip_special_tokens=True,
    #                                 clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()

import re
# 这个函数的目的是抽取True or False
def extract_last_num(text: str) -> float:

    res = re.findall(r"\b(True|False)\b", text)  # 匹配 123456.789
    if len(res) > 0:
        return res[-1]
    else:
        return False

def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels, max_words = 2500):
    all_count = 0
    count = 0
    index = 0
    all_samples, corrects, wrongs = [], [], []

    gd_list = []
    ans_list = []

    # for i in range(0, len(prompts), 2):
    for prompt, label in zip(prompts, labels):
        
        index += 1
        
        # check if the prompt length more than threshold
        promt_words = prompt.split(' ')
        prompt_words_num = len(promt_words)
        if(prompt_words_num >= max_words):
            cut_off_num = prompt_words_num - max_words

            # claim is necessary for fake news detection, but evidence could be cut off to some extendt.
            split_sentence = prompt.split('Evaluate the following assertion:')
            
            evidence = split_sentence[0]
            evidence_words = evidence.split(' ')
            cut_off_evidence = " ".join(evidence_words[:-1*cut_off_num])
            
            claim = split_sentence[1]
            prompt = cut_off_evidence + '. ' + 'Evaluate the following assertion:' +  claim
        

        inputs = tokenizer(prompt,  return_tensors="pt").to(device)

        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                do_sample=True,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        ans = extract_last_num(r_finetune_g[0])
        result = dict()
        # ans = extract_last_num(pro)
        gd = extract_last_num(label)
        # gd = extract_last_num(la)
        # print(ans, gd)
        result['index'] = index
        result['input'] = r_finetune_g[0]
        # result['input'] = pro
        result['label'] = label
        # result['label'] = la
        result['pred_bool'] = ans
        result['label_bool'] = gd
        if ans == gd:
            count += 1
            corrects.append(result)
        else:
            wrongs.append(result)
        
        all_samples.append(result)
        all_count += 1
        recall = fake_news_recall_computation(all_samples)
        print('TP+TN: ',count, ' All Samples: ', all_count, " Accuracy: ",count/all_count, ' Recall ', recall)
    
    
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
    #     json.dump(corrects, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
    #     json.dump(wrongs, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_generate_all.json", "w", encoding='utf-8') as f:
    #     json.dump(all_samples, f, ensure_ascii=False, indent=4)
    final_recall = fake_news_recall_computation(all_samples)
    print('Fake News Detection Results:')
    print('Total TP+TN: ',count, ' Total Samples: ', all_count, " Total Accuracy: ",count/all_count, ' Total Recall ', final_recall)
    print("====================prompt end=============================")
    print()
    print()
    return count/all_count, final_recall


def main():
    args = parse_args()

    device = torch.device("cuda:1")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_finetune,
                                  fast_tokenizer=True)
    # padding_side = 'right'
    tokenizer.pad_token=tokenizer.eos_token
#    model_baseline = create_hf_model(AutoModelForCausalLM,
#                                     args.model_name_or_path_baseline,
#                                     tokenizer, None)
    print('loading ckpt from ', args.model_name_or_path_finetune)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, bf16=True)

    model_fintuned.to(device)
    model_baseline = None

    results = {}
    
        
    prompt_no_input = (f"Below is an instruction that describes a fake news detection task. "
                       f'Write a response that appropriately completes the request.\n\n'
                       f'### Instruction:\n'
                       f'If there are only True and False categories, based on your knowledge and the '
                       "following information: {evidence} Evaluate the following assertion: {claim} If possible, "
                       f"please also give the reasons. \n\n### Response:."
                       )
        
    eval_path = f"/data1/haihongzhao/DSAA6000I-Final-Project-Group-7/data_for_LLM/data/test_use.jsonl"

    labels = []
    prompts = []
    fi = open(eval_path, encoding="utf8")
    for line in fi:
        line = line.strip()
        o = json.loads(line)
        prompt = prompt_no_input.format(evidence=o['evidence'],claim=o['claim']) 
        prompts.append(prompt)
        labels.append(o["response"])


    # prompts, labels = shuffle(np.array(prompts), np.array(labels))
    if args.batch_size == 1:
        total_acc, total_recall = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels)
    else:
        args.batch_size = 1
        total_acc, total_recall = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels)        
        
    import csv
    with open(Path(args.model_name_or_path_finetune) / f"Fake_News_Detection_RAWFC_and_LIAR_evaluate_results_bs{args.batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['"Fake_News_Detection', 'Accuracy', 'Recall'])
        writer.writerow(['RAWFC_and_LIAR', total_acc, total_recall])

    


if __name__ == "__main__":
    main()
