import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-svo", "--svo", type=str)
parser.add_argument("-o", "--output", type=str)
parser.add_argument("-oi", "--output_infer", type=str)

args = parser.parse_args()

with open(args.input, "r") as f:
    data = json.load(f)

with open(args.svo, "r") as f:
    svo_data = json.load(f)

outdir = os.path.dirname(args.output)
if not os.path.exists(outdir):
    os.makedirs(outdir)

out = []
out_infer =[]
count = 0
for i, d in enumerate(data):
    prompt = d["prompt"]
    svo = svo_data[i]
    if d["label"].strip() == "True":
        d["predicted_label"] = "False"
    else:
        d["predicted_label"] = "True"
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
        out_infer.append(data[i])

        count +=1 

with open(args.output, "w") as f:
    json.dump(out, f, indent=4)

with open(args.output_infer, "w") as f:
    json.dump(out_infer, f, indent=4)
