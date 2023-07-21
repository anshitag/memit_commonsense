"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from typing import List
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn import metrics

from util.generate import generate_fast
from util.perplexity import perplexity


def compute_rewrite_quality_cs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    noise_token: str,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in [noise_token, "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    # paraphrase_prompts = record["paraphrase_prompts"]
    # neighborhood_prompts = record["neighborhood_prompts"]
    # generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        # paraphrase_prompts,
        # neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        # [0 for _ in range(len(paraphrase_prompts))],
        # [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                # "paraphrase_prompts",
                # "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                # "paraphrase_prompts",
                # "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"][noise_token]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()


class CommonSenseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tok: AutoTokenizer,
        eval_dataset_type: str,
        dict_label: dict,
        sample_indexes: List[int] = None,
        max_length: int = 32,
    ):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        if sample_indexes is not None:
            samples = []
            for idx in sample_indexes:
                samples.append(self.data[idx])
            self.data = samples
            print("Eval", self.data)

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for e in self.data:
            if eval_dataset_type == 'efficacy':
                prompt = e["prompt"] + ":"
                tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=False)
                self.input_ids.append(torch.tensor(tok_output['input_ids']))
                self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
                self.labels.append(e["label"])
            else:
                for i in e[eval_dataset_type]:
                    prompt = i + ":"
                    tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=False)
                    self.input_ids.append(torch.tensor(tok_output['input_ids']))
                    self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
                    if eval_dataset_type == "affected_reasoning":
                        self.labels.append(dict_label[1])
                    else:
                        self.labels.append(e["label"])

        print(f"Loaded {eval_dataset_type} dataset with {len(self)} elements")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item]

    def getlabel(self):
        return self.labels
    
    def getdata(self):
        return self.data


def next_word_prediction(
    model: AutoModelForCausalLM,
    dataloader: DataLoader, 
    tok: AutoTokenizer, 
    device: str
):
    pred_label = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            colon_indexes = batch['attention_mask'].sum(dim = -1) - 1
            colon_indexes = colon_indexes.view(-1, 1)
            out = model(**batch).logits.argmax(dim=-1)
            prediction = torch.gather(out, 1, colon_indexes)
            pred_label += tok.batch_decode(prediction)
    return pred_label


def calculate_metrics(
    orig_label: List[str], 
    pred_label: List[str],
    eval_type: str,
    labels: List[str]
):
    acc = metrics.accuracy_score(orig_label, pred_label) * 100
    f1 = metrics.f1_score(orig_label, pred_label, average='weighted', labels=labels, zero_division=1) * 100
    print(f'{eval_type} Accuracy = ', acc)
    print(f'{eval_type} F1 score = ', f1)
    try:
        print(f'{eval_type} Confusion Matrix = \n', metrics.confusion_matrix(orig_label, pred_label, labels=labels))
    except:
        print('Confusion matrix cannot be calculated')
    print('Classification Report: \n',  metrics.classification_report(orig_label, pred_label, labels=labels, zero_division=1))
    return {
        eval_type+"_accuracy": acc, 
        eval_type+"_f1Score": f1
        }


def evaluate(
    model: AutoModelForCausalLM, 
    tok: AutoTokenizer, 
    data_file: str, 
    dict_label: dict,
    model_type: str, 
    inference_type: str, 
    device: str,
    sample_indexes: List[int]
):
    model.eval()

    if inference_type=='config':
        dataset = CommonSenseDataset(data_file, tok, 'efficacy', dict_label)

        data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data])}

        dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

        p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

        pred_label = next_word_prediction(model, dataloader, tok, device)
        true_pred = [dict_label[i] for i in dataset.getlabel()]
        prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
        split_metrics = calculate_metrics(true_pred, prediction_label, model_type, [p_index, n_index, other_index])

        output_data = []
        for idx, item in enumerate(dataset.getdata()):
            output_data.append({"prompt": item["prompt"], "label": dict_label[item["label"]], "predicted_label":prediction_label[idx]})


    else:
        eval_types = ['efficacy', 'unaffected_neighborhood_subject', 'unaffected_neighborhood_object', 'affected_neighborhood_subject', 'affected_neighborhood_object', 'affected_neighborhood_verb', 'affected_paraphrase', 'affected_reasoning']

        output_data = {}
        split_metrics = {}
        for e_type in eval_types:
            dataset = CommonSenseDataset(data_file, tok, e_type, dict_label, sample_indexes)

            data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                        'attention_mask': torch.stack([f[1] for f in data])}

            dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

            p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

            pred_label = next_word_prediction(model, dataloader, tok, device)
            prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]

            if e_type in ['unaffected_neighborhood_subject', 'unaffected_neighborhood_object']:
                output_data[e_type] = prediction_label
            else:
                true_pred = dataset.getlabel()
                split_metrics.update(calculate_metrics(true_pred, prediction_label, f'{model_type}_{e_type}', [p_index, n_index, other_index]))

    return split_metrics, output_data


def evaluate_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer, 
    inference_file: str, 
    model_type: str,
    inference_type: str,
    sample_indexes: List[int] = None
):
    dict_label = {1: " True", 0: " False"}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_metrics, output_data = evaluate(model, tok, inference_file, dict_label, model_type, inference_type, device, sample_indexes)
    return eval_metrics, output_data


def compare_unaffected_output(
    unedited_output: dict, 
    edited_output: dict
):
    dict_label = {1: " True", 0: " False"}
    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"
    split_metrics = {}

    for e_type in ['unaffected_neighborhood_subject', 'unaffected_neighborhood_object']:
        true_label = unedited_output[e_type]
        prediction_label = edited_output[e_type]
        split_metrics.update(calculate_metrics(true_label, prediction_label, e_type, [p_index, n_index, other_index]))
    return split_metrics


def compare_models(
    unedited_output: dict, 
    edited_output: dict
):
    unedited_inc = 0
    unedited_cor = 0
    for i in unedited_output:
        if i["predicted_label"]!= i["label"]:
            unedited_inc +=1
        else:
            unedited_cor +=1

    print(f"Unedited Pred: Incorrect {unedited_inc}, Correct {unedited_cor}")

    inc = 0
    cor = 0
    inc_changed = 0
    cor_changed = 0

    for i in range(0,len(unedited_output)):
        if unedited_output[i]["predicted_label"]!= unedited_output[i]["label"]:
            if edited_output[i]["predicted_label"]!=edited_output[i]["label"]:
                inc +=1
            else:
                cor_changed +=1
        else:
            if edited_output[i]["predicted_label"]!=edited_output[i]["label"]:
                inc_changed +=1
            else:
                cor +=1

    edited_cor = cor_changed*100/unedited_inc
    changed_inc = inc_changed*100/unedited_cor

    print(f"\nCompare Unedited Incorrect Predictions: \nRemained Incorrect {inc} \nChanged to Correct i.e. Efficacy {cor_changed} = {edited_cor:.2f}%")

    print(f"\nCompare Unedited Correct Predictions: \nChanged to Incorrect i.e. Relapse {inc_changed} = {changed_inc:.2f}% \nRemained Correct {cor}")

    return {
        "efficacy" : edited_cor,
        "relapse" : changed_inc
    }
