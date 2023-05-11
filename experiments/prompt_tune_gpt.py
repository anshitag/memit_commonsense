import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import json
import torch
from pathlib import Path
from sklearn import metrics

class CommonSenseDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        zero_shot_prompt: str,
        tok: AutoTokenizer,
    ):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for i in self.data:
        
            prompt = i["prompt"] + ". " + zero_shot_prompt 

            tok_output = tok(prompt, truncation=False)
            self.input_ids.append(torch.tensor(tok_output['input_ids']))
            self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
            self.labels.append(i["label"])

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item]

    def getlabel(self):
        return self.labels
    
    def getdata(self):
        return self.data

def calculate_metrics(orig_label, pred_label, split, labels):

    acc = metrics.accuracy_score(orig_label, pred_label)
    f1 = metrics.f1_score(orig_label, pred_label, average='weighted', labels=labels, zero_division=1)
    print(f'{split} Accuracy = ', acc)
    print(f'{split} F1 score = ', f1)
    try:
        print(f'{split} Confusion Matrix = \n', metrics.confusion_matrix(orig_label, pred_label, labels=labels))
    except:
        print('Confusion matrix cannot be calculated')
    print('Classification Report: \n',  metrics.classification_report(orig_label, pred_label, labels=labels, zero_division=1))
    return {split+"_accuracy": acc, split+"_f1Score": f1}

def boolean_prediction(model, dataloader, tok, dict_label, p_true_domain, p_false_domain):
    pred_list = []
    pred_label = []
    true_token = tok.encode(dict_label[1])
    false_token = tok.encode(dict_label[0])
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}

            output = model(**batch).logits
            
            probs = torch.softmax(output[:, -1], dim=1)
            p_true = probs[:, true_token]/p_true_domain
            p_false = probs[:, false_token]/p_false_domain

            max_pred = torch.max(p_true, p_false)
            prediction_bool = torch.eq(max_pred, p_true)
            prediction = torch.where(prediction_bool, true_token[0], false_token[0])

            pred_list += prediction.tolist()
            pred_label += tok.batch_decode(prediction)
    return pred_list, pred_label

def calculate_pmi(model, tok, zero_shot_prompt, dict_label):
    inputs = tok(zero_shot_prompt, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    output = model(**inputs).logits

    true_token = tok.encode(dict_label[1])
    false_token = tok.encode(dict_label[0])

    probs = torch.softmax(output[:,-1], dim=1)
    p_true = probs[:, true_token]
    p_false = probs[:, false_token]

    return p_true[0].item(), p_false[0].item()


def evaluate(MODEL, DS, DATATYPE, BATCH_SIZE, TEST_DATA_DIR, SPLIT, PMI, zero_shot_prompt, dict_label, device):

    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="left")
    tok.pad_token = tok.eos_token

    dataset = CommonSenseDataset(TEST_DATA_DIR, zero_shot_prompt, tok)

    data_collator = lambda data: { 'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data])}

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    p_true_domain, p_false_domain = 1, 1
    if PMI:
        p_true_domain, p_false_domain = calculate_pmi(model, tok, zero_shot_prompt, dict_label)
        print(f"P(1) = {p_true_domain} and P(0) = {p_false_domain} for domain: {zero_shot_prompt}")

    prediction, pred_label = boolean_prediction(model, train_dataloader, tok, dict_label, p_true_domain, p_false_domain)

    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

    true_pred = [dict_label[i] for i in dataset.getlabel()]
    prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
    split_metrics = calculate_metrics(true_pred, prediction_label, SPLIT, [p_index, n_index, other_index])

    output_data = []
    for idx, item in enumerate(dataset.getdata()):
        output_data.append({"prompt": item["prompt"], "label": dict_label[item["label"]], "predicted_label":pred_label[idx]})

    pmi_name = '_pmi' if PMI else ''
    with open(f'results/outputs/zero_shot_{MODEL}_{DS}_{DATATYPE}_{SPLIT}{pmi_name}.json', 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt2-xl')
    parser.add_argument('-d', '--dataset', type=str, default='20q')
    parser.add_argument('--datatype', type=str, default='normal')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('-b', '--log2_batch_size', type=int, default=0)
    parser.add_argument('--pmi', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dict_label = {1: " True", 0: " False"}

    BATCH_SIZE = pow(2, args.log2_batch_size)
    MODEL = args.model
    DS = args.dataset
    DATATYPE = args.datatype 
    SPLIT = args.split
    PMI = args.pmi

    zero_shot_prompt = "True or False?"

    combined_name = '_' + DATATYPE if DATATYPE != 'normal' else ''

    TEST_DATA_DIR = f"../commonsense_data/{DS}/{SPLIT}{combined_name}.json"

    evaluate(MODEL, DS, DATATYPE, BATCH_SIZE, TEST_DATA_DIR, SPLIT, PMI, zero_shot_prompt, dict_label, device)