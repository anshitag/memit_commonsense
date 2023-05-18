import argparse, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from sklearn import metrics
import numpy as np

class CommonSenseDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        tok: AutoTokenizer,
        dataset_type: str,
        dict_label: dict,
        max_length: int = 16
    ):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for i in self.data:
            if dataset_type == 'training':
                prompt = i["prompt"] + ":" + dict_label[i["label"]]
            else:
                prompt = i["prompt"] + ":"
            
            tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=True)
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


def next_word_prediction(model, dataloader, tok, true_vocab_index):

    pred_list = []
    pred_label = []
    true_prob_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            colon_indexes = batch['attention_mask'].sum(dim = -1) - 1
            colon_indexes = colon_indexes.view(-1, 1)
            out = model(**batch).logits
            prediction = torch.gather(out.argmax(dim=-1), 1, colon_indexes)
            pred_list += prediction.tolist()
            pred_label += tok.batch_decode(prediction)
            true_softmax = torch.softmax(out, dim=-1)[:,:,true_vocab_index]
            true_prob = torch.gather(true_softmax, 1, colon_indexes)
            true_prob_list += true_prob.tolist()
    return pred_list, pred_label, true_prob_list

def calculate_metrics(orig_label, pred_label, true_prob_list, labels):
    acc = metrics.accuracy_score(orig_label, pred_label)
    f1 = metrics.f1_score(orig_label, pred_label, average='weighted', labels=labels, zero_division=1)
    text_to_label_dict = {" True": 1, " False": 0}
    auc = metrics.roc_auc_score([text_to_label_dict[i] for i in orig_label], true_prob_list)
    print(f'Accuracy = {acc}')
    print(f'F1 score = {f1}')
    print(f'AUC = {auc}')
    try:
        print(f'Confusion Matrix = \n', metrics.confusion_matrix(orig_label, pred_label, labels=labels))
    except:
        print('Confusion matrix cannot be calculated')
    print('Classification Report: \n',  metrics.classification_report(orig_label, pred_label, labels=labels, zero_division=1))
    return {"accuracy": acc, "f1Score": f1, "auc": auc}


def evaluate(model, MODEL, DATA_FILE, BATCH_SIZE, dict_label, device, OUTPUT_FILE):

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token

    model.eval()

    dataset = CommonSenseDataset(DATA_FILE, tok, 'evaluation', dict_label)

    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[0] for f in data])}

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"
    true_vocab_index = tok.convert_tokens_to_ids(p_index.strip())

    prediction, pred_label, true_prob_list = next_word_prediction(model, dataloader, tok, true_vocab_index)
    true_pred = [dict_label[i] for i in dataset.getlabel()]
    prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
    split_metrics = calculate_metrics(true_pred, prediction_label, true_prob_list, [p_index, n_index, other_index])

    if OUTPUT_FILE:
        output_data = []
        for idx, item in enumerate(dataset.getdata()):
            output_data.append({"prompt": item["prompt"], "label": dict_label[item["label"]], "predicted_label":pred_label[idx]})
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt2-large')
    parser.add_argument('-mp', '--model_path', type=str, default='finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/peach-durian-1561')
    parser.add_argument('-i', '--input_file', type=str, default='commonsense_data/pep3k/test.json')
    parser.add_argument('-o', '--output_file', type=str, default=None)
    parser.add_argument('-b', '--log2_batch_size', type=int, default=6)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_to_text_dict = {1: " True", 0: " False"}
    MODEL = args.model
    BATCH_SIZE = pow(2, args.log2_batch_size)
    MODEL_PATH = args.model_path
    DATA_FILE = args.input_file
    OUTPUT_FILE = None
    if args.output_file:
        OUTPUT_FILE = args.output_file

    model = None
    try: 
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    except:
        print(f"No saved model found at this path: {MODEL_PATH}")
        exit()

    evaluate(model, MODEL, DATA_FILE, BATCH_SIZE, label_to_text_dict, device, OUTPUT_FILE)