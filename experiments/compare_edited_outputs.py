import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--unedited_prediction_output", type=str)
parser.add_argument("-e", "--edited_prediction_output", type=str)

args = parser.parse_args()

with open(args.unedited_prediction_output, "r") as f:
    old_pred = json.load(f)

with open(args.edited_prediction_output, "r") as f:
    new_pred = json.load(f)

inc = 0
cor = 0
for i in old_pred:
    if i["predicted_label"]!= i["label"]:
        inc +=1
    else:
        cor +=1

print(f"Unedited Pred: Incorrect {inc}, Correct {cor}")

inc = 0
cor = 0
for i in new_pred:
    if i["predicted_label"]!= i["label"]:
        inc +=1
    else:
        cor +=1
print(f"Edited Pred: Incorrect {inc}, Correct {cor}")

inc = 0
cor = 0
inc_changed = 0
cor_changed = 0
true_inc_changed = 0
false_inc_changed = 0
true_cor_changed = 0
false_cor_changed = 0

for i in range(0,len(old_pred)):
    if old_pred[i]["predicted_label"]!= old_pred[i]["label"]:
        if new_pred[i]["predicted_label"]!=new_pred[i]["label"]:
            inc +=1
        else:
            cor_changed +=1
            if new_pred[i]["label"]==" True":
                true_cor_changed +=1
            else:
                false_cor_changed +=1
    else:
        if new_pred[i]["predicted_label"]!=new_pred[i]["label"]:
            inc_changed +=1
            if new_pred[i]["label"]==" True":
                true_inc_changed +=1
            else:
                false_inc_changed +=1
        else:
            cor +=1
print(f"\nCompare Unedited Incorrect Predictions: \nRemained Incorrect {inc} \nChanged to Correct {cor_changed}")
print(f"Correctly changed to True {true_cor_changed} = {true_cor_changed*100/cor_changed:.2f}% and False: {false_cor_changed} = {false_cor_changed*100/cor_changed:.2f}%")

print(f"\nCompare Unedited Correct Predictions: \nChanged to Incorrect {inc_changed} \nRemained Correct {cor}")
print(f"Incorrectly changed to True {true_inc_changed} = {true_inc_changed*100/inc_changed:.2f}% and False: {false_inc_changed} = {false_inc_changed*100/inc_changed:.2f}%")
