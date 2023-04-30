import argparse, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed, TrainerCallback
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from sklearn import metrics
import numpy as np
import wandb

class CommonSenseDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        tok: AutoTokenizer,
        dict_label: dict,
        max_length: int = 16
    ):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        self.input_ids = []
        self.attention_mask = []
        self.target_labels = []
        self.labels = []

        for i in self.data:
            prompt = i["prompt"] + ":"
            tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=True)
            self.input_ids.append(torch.tensor(tok_output['input_ids']))
            self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
            label_index = tok(dict_label[i["label"]], return_tensors='pt')['input_ids'][0][0]
            self.target_labels.append(label_index)
            self.labels.append(i["label"])

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.target_labels[item]

    def getlabel(self):
        return self.labels
    
    def get_input_ids(self):
        return self.input_ids
    
    def getdata(self):
        return self.data

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        colon_indexes = attention_mask.sum(dim = -1) - 1
        target_labels = inputs['labels']
        colon_indexes = colon_indexes.view(-1, 1, 1)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        vocab_size = logits.size(2)
        colon_indexes = colon_indexes.repeat_interleave(repeats=vocab_size, dim=2)
        logits = torch.gather(logits, 1, colon_indexes).squeeze(1)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, target_labels)
        return (loss, out) if return_outputs else loss
    

def next_word_prediction(model, dataloader, tok):

    pred_list = []
    pred_label = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            colon_indexes = batch['attention_mask'].sum(dim = -1) - 1
            colon_indexes = colon_indexes.view(-1, 1)
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.argmax(dim=-1)
            prediction = torch.gather(out, 1, colon_indexes)
            pred_list += prediction.tolist()
            pred_label += tok.batch_decode(prediction)
    return pred_list, pred_label


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


def train(MODEL, CHECKPOINT, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, SAVE_MODEL, dict_label, device):

    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token

    wandb.watch(model)

    train_dataset = CommonSenseDataset(TRAIN_DATA_FILE, tok, dict_label)
    valid_dataset = CommonSenseDataset(VALID_DATA_FILE, tok, dict_label)

    checkpoint_path = f'result/checkpoints/{DS}/{MODEL}/{DATATYPE}/{wandb.run.name}'

    training_args = TrainingArguments(output_dir=checkpoint_path, 
                                num_train_epochs=EPOCHS,
                                save_total_limit = 2,
                                load_best_model_at_end=True, 
                                save_strategy="epoch", 
                                evaluation_strategy="epoch",
                                logging_strategy="epoch",
                                per_device_train_batch_size=BATCH_SIZE, 
                                per_device_eval_batch_size=BATCH_SIZE,  
                                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
                                weight_decay=0.01,
                                # disable_tqdm=True,
                                learning_rate=LEARNING_RATE,
                                metric_for_best_model='f1_score',
                                report_to="wandb")

    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[2] for f in data])}
    
    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"
    possible_labels = [p_index, n_index, other_index]
    true_pred = [dict_label[i] for i in valid_dataset.getlabel()]

    def compute_metrics(p):
        logits, _ = p
        input_ids = torch.stack(valid_dataset.get_input_ids())
        input_ids = np.array(input_ids)
        colon_indexes = (input_ids != tok.pad_token_id).sum(axis = -1) - 1
        colon_indexes = colon_indexes.reshape(-1, 1, 1)
        logits = np.take_along_axis(logits, colon_indexes, 1).squeeze(1)
        predictions = tok.batch_decode(logits.argmax(axis=-1).flatten())
        predictions = [other_index if i not in dict_label.values() else i for i in predictions]   
        try:
            confusion_matrix = metrics.confusion_matrix(true_pred, predictions, labels=possible_labels)
        except:
            confusion_matrix = None
        print('Confusion Matrix: ', confusion_matrix)
        return {
                'accuracy': metrics.accuracy_score(true_pred, predictions),
                'f1_score': metrics.f1_score(true_pred, predictions, average='weighted', labels=possible_labels, zero_division=1)
        }

    # start training
    trainer = CustomTrainer(model=model, 
                args=training_args, 
                train_dataset=train_dataset, 
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])


    trainer.train()
    print(trainer.evaluate())

    if SAVE_MODEL:
        trainer.save_model(output_dir=f'result/best-checkpoints/{DS}/{MODEL}/{DATATYPE}')

    os.system(f"rm -rf {checkpoint_path}")
    return model


def evaluate(model, MODEL, DS, DATATYPE, RUN_NAME, TRAIN_DATA_FILE, VALID_DATA_FILE, BATCH_SIZE, dict_label, device):
    if not model:
        try: 
            # model = AutoModelForCausalLM.from_pretrained(f'result/best-checkpoints/pep3k/gpt2-large/normal/').to(device)
            model = AutoModelForCausalLM.from_pretrained(f'result/best-checkpoints/{DS}/{MODEL}/{DATATYPE}/{RUN_NAME}').to(device)
        except:
            print("No saved model found for this run. " + f'result/best-checkpoints/{DS}/{MODEL}/{DATATYPE}/{RUN_NAME}')
            exit()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token

    model.eval()

    train_dataset = CommonSenseDataset(TRAIN_DATA_FILE, tok, dict_label)
    valid_dataset = CommonSenseDataset(VALID_DATA_FILE, tok, dict_label)

    data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[2] for f in data])}

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

    for split, dataloader, dataset in [('train', train_dataloader, train_dataset), ('valid', valid_dataloader, valid_dataset)]:
        _, pred_label = next_word_prediction(model, dataloader, tok)
        true_pred = [dict_label[i] for i in dataset.getlabel()]
        prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
        split_metrics = calculate_metrics(true_pred, prediction_label, split, [p_index, n_index, other_index])

        wandb.log(split_metrics)

        if split=='valid':
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
    parser.add_argument('-b', '--log2_batch_size', type=int, default=6)
    parser.add_argument('-g', '--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--wandb_account', type=str, default="anshitag")
    parser.add_argument('--wandb_project', type=str, default="memit_commonsense_valid_finetune")
    parser.add_argument('--run_name', type=str, default=None) # Specify when evaluating
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_account, config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dict_label = {1: " True", 0: " False"}
    BATCH_SIZE = pow(2, args.log2_batch_size)
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    EPOCHS = args.epochs
    MODEL = args.model
    CHECKPOINT = args.checkpoint
    DS = args.dataset
    DATATYPE = args.datatype 
    LEARNING_RATE = args.learning_rate
    RUN_NAME = args.run_name
    SAVE_MODEL = args.save_model

    combined_name = '_' + DATATYPE if DATATYPE != 'normal' else ''

    TRAIN_DATA_FILE = f"incorrect-data/{MODEL}_{DS}_{DATATYPE}_valid.json"
    VALID_DATA_FILE = f"../../../commonsense_data/{DS}/valid{combined_name}.json"
    
    model = None
    if not args.evaluation_only:
        model = train(MODEL, CHECKPOINT, DS, DATATYPE, TRAIN_DATA_FILE, VALID_DATA_FILE, EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, SAVE_MODEL, dict_label, device)
    evaluate(model, MODEL, DS, DATATYPE, RUN_NAME, TRAIN_DATA_FILE, VALID_DATA_FILE, BATCH_SIZE, dict_label, device)