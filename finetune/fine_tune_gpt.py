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


def next_word_prediction(model, dataloader, tok):

    pred_list = []
    pred_label = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch).logits
            prediction = torch.argmax(out[:, -1], dim=1)
            pred_list += prediction.tolist()
            pred_label += tok.batch_decode(prediction)
    return pred_list, pred_label


def calculate_metrics(orig_label, pred_label, split, labels):

    print(f'{split} Accuracy = ', metrics.accuracy_score(orig_label, pred_label))
    print(f'{split} F1 score = ', metrics.f1_score(orig_label, pred_label, average='macro', labels=labels, zero_division=1))
    try:
        print(f'{split} Confusion Matrix = \n', metrics.confusion_matrix(orig_label, pred_label, labels=labels))
    except:
        print('Confusion matrix cannot be calculated')
    print('Classification Report: \n',  metrics.classification_report(orig_label, pred_label, labels=labels, zero_division=1))


def train(MODEL, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, EPOCHS, BATCH_SIZE, dict_label, device):

    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="left")
    tok.pad_token = tok.eos_token

    train_dataset = CommonSenseDataset(TRAIN_DATA_FILE, tok, 'training', dict_label)
    valid_dataset = CommonSenseDataset(VALID_DATA_FILE, tok, 'training', dict_label)

    training_args = TrainingArguments(output_dir=f'result/checkpoints/{DS}/{MODEL}/{DATATYPE}', 
                                num_train_epochs=EPOCHS,
                                save_total_limit = 2,
                                # load_best_model_at_end=True, 
                                save_strategy="epoch", 
                                evaluation_strategy="epoch",
                                per_device_train_batch_size=BATCH_SIZE, 
                                per_device_eval_batch_size=BATCH_SIZE,
                                weight_decay=0.01,
                                disable_tqdm=True)

    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[0] for f in data])}
    # start training
    trainer = Trainer(model=model, 
                args=training_args, 
                train_dataset=train_dataset, 
                eval_dataset=valid_dataset,
                data_collator=data_collator)

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(output_dir=f'result/best-checkpoints/{DS}/{MODEL}/{DATATYPE}')


def evaluate(MODEL, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, TEST_DATA_FILE, BATCH_SIZE, dict_label, device):

    model = AutoModelForCausalLM.from_pretrained(f'result/best-checkpoints/{DS}/{MODEL}/{DATATYPE}').to(device)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="left")
    tok.pad_token = tok.eos_token

    model.eval()

    train_dataset = CommonSenseDataset(TRAIN_DATA_FILE, tok, 'evaluation', dict_label)
    valid_dataset = CommonSenseDataset(VALID_DATA_FILE, tok, 'evaluation', dict_label)
    test_dataset = CommonSenseDataset(TEST_DATA_FILE, tok, 'evaluation', dict_label)

    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[0] for f in data])}

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

    for split, dataloader, dataset in [('train', train_dataloader, train_dataset), ('valid', valid_dataloader, valid_dataset), ('test', test_dataloader, test_dataset)]:
        prediction, pred_label = next_word_prediction(model, dataloader, tok)
        true_pred = [dict_label[i] for i in dataset.getlabel()]
        prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
        calculate_metrics(true_pred, prediction_label, split, [p_index, n_index, other_index])

        if split=='test':
            output_data = []
            for idx, item in enumerate(dataset.getdata()):
                output_data.append({"prompt": item["prompt"], "label": dict_label[item["label"]], "predicted_label":pred_label[idx]})
            with open(f'result/outputs/{MODEL}_{DS}_{DATATYPE}.json', 'w') as f:
                json.dump(output_data, f, indent=4)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt2-medium')
    parser.add_argument('-d', '--dataset', type=str, default='20q')
    parser.add_argument('--datatype', type=str, default='normal')
    parser.add_argument('-e', '--evaluation_only', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-ep', '--epochs', type=int, default=2)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dict_label = {1: " True", 0: " False"}
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = args.model
    DS = args.dataset
    DATATYPE = args.datatype 

    combined_name = '_' + DATATYPE if DATATYPE != 'normal' else ''

    TRAIN_DATA_FILE = f"../commonsense_data/{DS}/train{combined_name}.json"
    VALID_DATA_FILE = f"../commonsense_data/{DS}/valid{combined_name}.json"
    TEST_DATA_FILE = f"../commonsense_data/{DS}/test{combined_name}.json"

    if not args.evaluation_only:
        train(MODEL, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, EPOCHS, BATCH_SIZE, dict_label, device)
    evaluate(MODEL, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, TEST_DATA_FILE, BATCH_SIZE, dict_label, device)




