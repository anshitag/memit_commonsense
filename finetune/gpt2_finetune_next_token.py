"""## **Imports**

Import all needed libraries for this notebook.

Declare parameters used for this notebook:

* `set_seed(123)` - Always good to set a fixed seed for reproducibility.
* `epochs` - Number of training epochs (authors recommend between 2 and 4).
* `batch_size` - Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length a batch of 10 USUALY works without cuda memory issues. For small sequence length can try batch of 32 or higher.
max_length - Pad or truncate text sequences to a specific length. I will set it to 60 to speed up training.
* `device` - Look for gpu to use. Will use cpu by default if no gpu found.
* `model_name_or_path` - Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk. In this tutorial I will use `gpt2` model.
* `labels_ids` - Dictionary of labels and their id - this will be used to convert string labels to numbers.
* `n_labels` - How many labels are we using in this dataset. This is used to decide size of classification head.
"""

import io
import os
import torch
import re
import json
import argparse
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
# from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from copy import deepcopy
from transformers import (set_seed,
                        TrainingArguments,
                        Trainer,
                        GPT2Config,
                        GPT2Tokenizer,
                        TrainerCallback,
                        AdamW, 
                        get_linear_schedule_with_warmup,
                        AutoModelForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='gpt2-medium')
parser.add_argument('-d', '--dataset', default='20q')
parser.add_argument('--datatype', default='')
parser.add_argument('-p', '--positive', default='true')
parser.add_argument('-n', '--negative', default='false')
parser.add_argument('-t', '--test', action='store_true')
args = parser.parse_args()
  
# Set seed for reproducibility.
set_seed(123)

epochs = 40

batch_size = 32

max_length = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


pos_token, neg_token = args.positive, args.negative
label_mapping = {0: neg_token, 1: pos_token}


model_name_or_path = args.model

class CommonSenseDataset(Dataset):
    def __init__(self, texts, labels, use_tokenizer):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for i in range(len(texts)):
            text = texts[i]
            label = labels[i]
            prep_txt = f'<startoftext>{text}: {label_mapping[label]}<endoftext>'
            text_encoding = use_tokenizer(prep_txt, truncation=True, max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(text_encoding['input_ids']))
            self.attn_masks.append(torch.tensor(text_encoding['attention_mask']))
            self.labels.append(label)
        self.n_examples = len(self.labels)
        # print(self.input_ids[:10])
        # print(self.labels[:10])
        
        return

    def __len__(self):

        return self.n_examples

    def __getitem__(self, item):

        return self.input_ids[item], self.attn_masks[item], self.labels[item]



# Get model configuration.
print('Loading configuraiton...')
# model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
# # default to left padding
# tokenizer.padding_side = "left"
# # Define PAD Token = EOS Token = 50256
# tokenizer.pad_token = tokenizer.eos_token
print('Length tokenizer', len(tokenizer))

# Get the actual model.
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

eos_token_id = tokenizer(f'{tokenizer.eos_token}')["input_ids"][0]
p_index, n_index = tokenizer(f' {pos_token}')["input_ids"][0], tokenizer(f' {neg_token}')["input_ids"][0]

# fix model padding token id
# model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

# Create data collator to encode text and labels into numbers.

def load_json_data(filename: str):
    content = io.open(filename, mode='r', encoding='utf-8').read()
    data = json.loads(content)
    texts = [d['prompt'] for d in data]
    labels = [d['label'] for d in data]
    return texts, labels

print('Loading Train Dataset...')

combined_name = '_' + args.datatype if args.datatype else ''
texts_train, labels_train = load_json_data(f'../commonsense_data/{args.dataset}/train{combined_name}.json')

print(texts_train[:10])
print(labels_train[:10])

print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = CommonSenseDataset(texts=texts_train,
                                    labels=labels_train,
                                    use_tokenizer=tokenizer,
                                    )
print('Created `train_dataset` with %d examples!'%len(train_dataset))

print('Loading Validation Dataset...')

texts_valid, labels_valid = load_json_data(f'../commonsense_data/{args.dataset}/valid{combined_name}.json')

print(texts_valid[:10])
print(labels_valid[:10])

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  CommonSenseDataset(texts=texts_valid,
                                    labels=labels_valid,
                                    use_tokenizer=tokenizer
                                    )
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.



class CustomCallback(TrainerCallback):
        
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def compute_metrics(p):
    global eos_token_id
    predictions, labels = p
    predictions = predictions.argmax(axis = 2)
    max_indexes = labels == tokenizer.eos_token_id
    indexes = max_indexes.argmax(axis = 1)[:, None]
    predictions = np.take_along_axis(predictions, (indexes - 2), axis = 1).flatten()
    labels = np.take_along_axis(labels, (indexes - 1), axis = 1).flatten()
    try:
        confusion_matrix = metrics.confusion_matrix(predictions, labels, labels=[p_index, n_index])
    except:
        confusion_matrix = None
    print('Confusion Matrix: ', confusion_matrix)
    return {
            'accuracy': metrics.accuracy_score(predictions, labels),
            'f1_score': metrics.f1_score(predictions, labels, average='macro', labels=[p_index, n_index])
    }


data_collator=lambda data: { 'input_ids': torch.stack([f[0] for f in data]),
                            'attention_mask': torch.stack([f[1] for f in data]),
                            'labels': torch.stack([f[0] for f in data]) }

datatype_string = args.datatype or 'normal'
training_args = TrainingArguments(output_dir=f'checkpoints/{args.dataset}/{args.model}/{datatype_string}', num_train_epochs=5, save_total_limit = 2,load_best_model_at_end=True, save_strategy='steps',
                                                            evaluation_strategy='steps', metric_for_best_model='f1_score', eval_steps=100, save_steps=100, logging_steps = 100, eval_delay = 0,
                                                            per_device_train_batch_size=8, per_device_eval_batch_size=8, warmup_steps=0, weight_decay=0.00, logging_dir='fineTune/logs',
                                                            disable_tqdm=True)


trainer = Trainer(model=model, args=training_args, train_dataset = train_dataset, eval_dataset=valid_dataset, compute_metrics=compute_metrics,
                data_collator=data_collator)

if not args.test:
    # Training
    trainer.train()
    trainer.save_model(output_dir=f'best-checkpoints/{args.dataset}/{args.model}/{datatype_string}')
    # trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
else:
    model = AutoModelForCausalLM.from_pretrained(f'best-checkpoints/{args.dataset}/{args.model}/{datatype_string}')
    model.to(device)

# Testing
print('Loading Test Dataset...')

texts_test, labels_test = load_json_data(f'../commonsense_data/{args.dataset}/test{combined_name}.json')
print(texts_test[:10])
print(labels_test[:10])


print('Dealing with Test...')
test_dataset =  CommonSenseDataset(texts=texts_test,
                                labels=labels_test,
                                use_tokenizer=tokenizer
                                )
print('Created `test_dataset` with %d examples!'%len(test_dataset))


model.eval()


def get_metrics(dataloader, split='train'):
    orig_label, pred_label = [], []
    tot_loss = 0
    global model
    for batch in dataloader:
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():        
            outputs = model(**batch)
            loss, logits = outputs[:2]
            tot_loss += loss.item()
            
            # Move logits and labels to CPU

            predictions = logits.argmax(dim = -1).detach().cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            del logits
            # get predicitons to list
            
            max_indexes = labels == tokenizer.eos_token_id
            indexes = max_indexes.argmax(axis = 1)[:, None]
        
            predictions = np.take_along_axis(predictions, (indexes - 2), axis = 1).flatten()
            labels = np.take_along_axis(labels, (indexes - 1), axis = 1).flatten()

            # update list
            orig_label += predictions.tolist()
            pred_label += labels.tolist()

    print(f'{split} Loss = ', tot_loss / len(dataloader))
    print(f'{split} Accuracy = ', metrics.accuracy_score(orig_label, pred_label))
    print(f'{split} F1 score = ', metrics.f1_score(orig_label, pred_label, average='macro', labels=[p_index, n_index]))
    try:
        print(f'{split} Confusion Matrix = ', metrics.confusion_matrix(orig_label, pred_label, labels=[p_index, n_index]))
    except:
        print('Confusion matrix cannot be calculated')
    print('Classification Report',  metrics.classification_report(orig_label, pred_label, labels=[p_index, n_index]))


train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
get_metrics(train_dataloader, 'train')
del train_dataloader

valid_dataloader = DataLoader(valid_dataset, batch_size=8, collate_fn=data_collator)
get_metrics(valid_dataloader, 'valid')
del valid_dataloader

test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)
get_metrics(test_dataloader, 'test')
del test_dataloader

# print('Confusion Matrix = ', metrics.confusion_matrix(orig_label, pred_label, labels=[pos_token, neg_token]))
    # Calculate the average loss over the training data.

orig_label, pred_label, orig_text, pred_text = [], [], [], []

for text, label in zip(texts_test, labels_test):
    prompt = f'<startoftext>{text}:'
    input = tokenizer(prompt, return_tensors='pt')
    input = {k:v.type(torch.long).to(device) for k,v in input.items()}
    output = model(**input).logits[0, -1].argmax()
    predicted_label = tokenizer.decode(output)
    predicted_text = prompt + predicted_label
    orig_label.append(label_mapping[label])
    pred_label.append(predicted_label[1:])
    orig_text.append(text)
    pred_text.append(predicted_text)


df = pd.DataFrame({ 'original_text': orig_text, 'predicted_label': pred_label, 'original_label': orig_label, 'predicted_text': pred_text })
df.to_csv(f'results/{args.model}-{args.dataset}-{datatype_string}.csv')
print('Accuracy = ', metrics.accuracy_score(orig_label, pred_label))
print('F1 score = ', metrics.f1_score(orig_label, pred_label, average='macro', labels=['true', 'false']))
print('Confusion Matrix = ', metrics.confusion_matrix(orig_label, pred_label, labels=[pos_token, neg_token]))
        