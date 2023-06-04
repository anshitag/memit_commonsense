import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-svo", "--svo", type=str)
parser.add_argument("-o", "--output", type=str)

args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f)

with open(args.svo, "r") as f:
    svo_data = json.load(f)

outdir = os.path.dirname(args.output)
if not os.path.exists(outdir):
    os.makedirs(outdir)

out = []
count = 0
for i, d in enumerate(data):
    prompt = d["prompt"]
    # print(prompt, label, pred_label)
    if  d["predicted_label"] == d["label"]:
        continue
    svo = svo_data[i]
    if svo["subject"] and svo["verb"] and svo["object"]:
        out.append({
            "case_id": count,
            "requested_rewrite": {  "prompt": f"{prompt}:", 
                                    "target_new": {
                                        "str": d["label"].strip(),
                                    },
                                    "target_true": {
                                        "str": d["predicted_label"].strip(),
                                    },
                                    "subject": svo["subject"],
                                    "verb": svo["verb"],
                                    "object": svo["object"],
                                }
        })
        count +=1 

with open(args.output, "w") as f:
    json.dump(out, f, indent=4)