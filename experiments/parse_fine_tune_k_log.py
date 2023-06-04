import re
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
parser.add_argument('--max_k', type=int, default=9)
args = parser.parse_args()

with open(args.input, 'r') as f:
    log = f.read()

MAX_K = args.max_k
metrics = [dict({ "k": 2**k }) for k in range(MAX_K)]


matching_sequences = ['efficacy', 'affected_reasoning', 'affected_neighborhood_subject', 'affected_neighborhood_object', 'affected_neighborhood_verb', 'affected_paraphrase',
                      'unaffected_neighborhood_subject', 'unaffected_neighborhood_object']

for seq in matching_sequences:
    accs = re.findall(rf'^{seq} Accuracy =  (\d+\.\d+)$', log, re.MULTILINE)
    f1scores = re.findall(rf'^{seq} F1 score =  (\d+\.\d+)$', log, re.MULTILINE)
    accs = [float(n) for n in accs]
    f1scores = [float(n) for n in f1scores]
    for idx in range(MAX_K):
        metrics[idx][f'edited_{seq}_accuracy'] = accs[idx]
        metrics[idx][f'edited_{seq}_f1Score'] = f1scores[idx]

with open(args.output, 'w') as f:
    json.dump(metrics, f, indent=4)



