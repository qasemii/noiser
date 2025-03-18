import argparse
import os
import torch
import numpy as np
import random
import nltk
import pickle
import json
import re
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from util.globals import DATA_DIR
from importance_score_evaluator.utils import (
    check_whitespace,
    collect_token_range,
    match_tokens_with_scores
)

from dsets import (
    KnownsDataset,
    CounterFactDataset,
)

from rationalization.src.evaluation.evaluator.soft_norm_sufficiency import SoftNormalizedSufficiencyEvaluator
from rationalization.src.evaluation.evaluator.soft_norm_comprehensiveness import SoftNormalizedComprehensivenessEvaluator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

device = "cuda"

# random.seed(42)
# torch.manual_seed(42)
# torch.use_deterministic_algorithms(True, warn_only=True)


def predict_token(model, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    # The following only is True for OLMo models
    if 'token_type_ids' in inp.keys():
        inp.pop('token_type_ids')

    logits = model(**inp)["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=-1) 
    probs, preds = torch.max(probs, dim=-1, keepdim=True)  # Keep dims for consistency
    result = tokenizer.decode(preds.squeeze(0).item())
    return result


def main():
    parser = argparse.ArgumentParser(description="Rationalization")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="Qwen/Qwen2-0.5B")
    aa("--dataset", default="Knowns")
    aa("--output_dir", default=f"results/")
    aa("--n_samples", default=-1, type=int)
    aa("--max_new_tokens", default=1, type=int)
    aa("--norm", default='2')
    aa("--mode", default='prob')
    aa("--method", default="noiser", type=str,
       help="noiser, attention, attention_last, attention_rollout, gradient_shap,\
             input_x_gradient, integrated_gradients, lime, reagent, occlusion")  
    aa("--openai_api_key", type=str, default=None)
    aa("--topk", type=int, default=50)
    
    args = parser.parse_args()

    result_dir = f"{args.output_dir}{args.dataset}/{args.model_name}"
    os.makedirs(result_dir, exist_ok=True)

    cache_dir = f"cache/{args.model_name}"
    os.makedirs(cache_dir, exist_ok=True)

    print('Loading model and tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # model.config.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None: #This is for OLMo models
        tokenizer.bos_token_id = tokenizer.pad_token_id
        # model.config.bos_token_id = tokenizer.pad_token_id
    

    print(f"Loading {args.dataset} dataset ...")
    if args.dataset == "Knowns":
        dataset = KnownsDataset(DATA_DIR)
    elif args.dataset == "Counterfact":
        dataset = CounterFactDataset(DATA_DIR)
    elif args.dataset == "LongRA":
        with open("/content/noiser/data/LongRA.json", "r") as f:
            dataset = json.load(f)
    else:
        raise ValueError
    
    # Filter dataset to only include examples where the predicted token matches the target
    print(f"Filtering dataset ...")
    dataset = [
        d for d in dataset
        if predict_token(model, tokenizer, d['prompt']).strip() == d['target']
    ]
    print(f"Filtered dataset to {len(dataset)} examples")

    nltk.download('punkt_tab')

    if args.method == 'noiser':
        from importance_score_evaluator.noiser import  NoiserImportanceScoreEvaluator
        rationalizer = NoiserImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            norm=args.norm,
            mode=args.mode
        )
    elif args.method == 'random':
        pass
    elif args.method == 'attention_last' or args.method == 'attention_rollout':
        from importance_score_evaluator.attention import AttentionImportanceScoreEvaluator
        rationalizer = AttentionImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            attn_type=args.method.replace("attention_", "")
        )
    else: #['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap']
        # assert args.method in ['integrated_gradients', 'input_x_gradient', 'attention', 'gradient_shap'] # input_x_gradient = signed in self written
        from importance_score_evaluator.inseq import  InseqImportanceScoreEvaluator
        rationalizer = InseqImportanceScoreEvaluator(
            model=model,
            tokenizer=tokenizer,
            method=args.method, 
            attribute_params={}
        )


    client = OpenAI(
            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
    
    # INSTRUCTION = (
    #     "# Task:\n"
    #     "Given a set of words extracted from a prompt for a completion task, "
    #     "return top 5 words as the most probable completions for the unseen "
    #     "prompt without any explanation."
    # )

    INSTRUCTION = (
        "# Task:\n"
        "Given a set of words extracted from a prompt for a completion task, "
        "return only the list of top 5 words as the most probable completions for the unseen "
        "prompt WITHOUT any explanation."
    )


    PROMPT = (
        "Tokens: {tokens}\n"
        "Probable Completion: "
    )

    generation_configs = {
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 200,
    }
    
    answ_top1_rate = []
    answ_top1_score = []

    answ_top5_rate = []
    answ_top5_score = []
    print("Starting rationalization ...")

    samples = dataset if args.n_samples == -1 else random.choices(dataset, k=args.n_samples)
    for data in tqdm(samples):
        idx = data['id']

        input_ids = tokenizer(data["prompt"], return_tensors='pt')['input_ids'][0].to(device)
        attention_mask = tokenizer(data["prompt"], return_tensors='pt')['attention_mask'][0].to(device)
        generated_ids = model.generate(input_ids=torch.unsqueeze(input_ids, 0),
                                       attention_mask= torch.unsqueeze(attention_mask, 0),
                                       pad_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=args.max_new_tokens, 
                                       do_sample=False)[0]
        # Gemma and Llama add [bos] token which should be exclude from input prompt when
        for target_pos in torch.arange(input_ids.shape[0], generated_ids.shape[0]):
            input_ids = torch.unsqueeze(generated_ids[:target_pos], 0)
            target_id = torch.unsqueeze(generated_ids[target_pos], 0)

            # rationalization
            rationalizer.rationalize(input_ids, target_id)
            scores = rationalizer.mean_important_score.unsqueeze(0).to(device)

            input_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
            tokens = nltk.word_tokenize(input_text)
            tokens = ['"' if token in ['``', "''"] else token for token in tokens]
            tokens = check_whitespace(input_text, tokens)
            tokens_range = collect_token_range(tokenizer, input_text, tokens)            
            scores = match_tokens_with_scores(scores.squeeze(), tokens_range)

            k = int((args.topk/100) * len(tokens))
            topk_indices = torch.topk(scores, k=k).indices.sort().values
            topk_words = [tokens[i.item()] for i in topk_indices]
            topk_scores = torch.gather(scores, 0, topk_indices)

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "system", "content": INSTRUCTION},
                          {"role": "user",  "content": PROMPT.format(tokens=topk_words)}],
                **generation_configs
            )

            prediction = response.choices[0].message.content

            ## Uncomment the following if using GPT-4o
            # prediction = [word.strip(" '") for word in prediction.split(",")]
            ## Uncomment the following if using meta-llama/Llama-3.3-70B-Instruct-Turbo
            prediction = re.findall(r'\b[A-Za-z]+\b', prediction)

            try:
                # Top-1
                if prediction[0] == data["target"]:
                    answ_top1_rate.append(1.0)
                    answ_top1_score.append(torch.sum(topk_scores).item())
                else:
                    answ_top1_rate.append(0.0)
                    answ_top1_score.append(0.0)

                # Top-5
                if data["target"] in prediction:
                    answ_top5_rate.append(1.0)
                    answ_top5_score.append(torch.sum(topk_scores).item())
                else:
                    answ_top5_rate.append(0.0)
                    answ_top5_score.append(0.0)
            except:
                pass

            # # compute metrics on Soft-NS and Soft-NC
            # print(f"Prompt: {data['prompt']}")
            # print(f"tokens: {topk_words}")
            # print(f"scores: {topk_scores}")
            # print(f'GPT prediction: {prediction}')
            # print("-"*10)
    print()
    print(f"Rate: {torch.mean(torch.tensor(answ_top1_rate, dtype=torch.float)).item()}")
    print(f"Score: {torch.mean(torch.tensor(answ_top1_score, dtype=torch.float)).item()}")
    print()
    print(f"Rate: {torch.mean(torch.tensor(answ_top5_rate, dtype=torch.float)).item()}")
    print(f"Score: {torch.mean(torch.tensor(answ_top5_score, dtype=torch.float)).item()}")
    



if __name__ == "__main__":
    main()



