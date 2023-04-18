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

# {
#     "case_id": 0,
#     "pararel_idx": 2796,
#     "requested_rewrite": {
#       "prompt": "The mother tongue of {} is",
#       "relation_id": "P103",
#       "target_new": {
#         "str": "English",
#         "id": "Q1860"
#       },
#       "target_true": {
#         "str": "French",
#         "id": "Q150"
#       },
#       "subject": "Danielle Darrieux"
#     },
#     "paraphrase_prompts": [
#       "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native",
#       "An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language"
#     ],
#     "neighborhood_prompts": [
#       "The mother tongue of L\u00e9on Blum is",
#       "The native language of Montesquieu is",
#       "Fran\u00e7ois Bayrou, a native",
#       "The native language of Raymond Barre is",
#       "Michel Rocard is a native speaker of",
#       "Jacques Chaban-Delmas is a native speaker of",
#       "The native language of Fran\u00e7ois Bayrou is",
#       "Maurice Genevoix, speaker of",
#       "The mother tongue of Fran\u00e7ois Bayrou is",
#       "Melchior de Vog\u00fc\u00e9, speaker of"
#     ],
#     "attribute_prompts": [
#       "J.\u00a0R.\u00a0R. Tolkien is a native speaker of",
#       "The mother tongue of Douglas Adams is",
#       "The mother tongue of Paul McCartney is",
#       "Elvis Presley is a native speaker of",
#       "Barack Obama, speaker of",
#       "Douglas Adams, speaker of",
#       "Meryl Streep, a native",
#       "George Orwell spoke the language",
#       "George Washington, a native",
#       "Michael Jackson, a native"
#     ],
#     "generation_prompts": [
#       "Danielle Darrieux"s mother tongue is",
#       "Where Danielle Darrieux is from, people speak the language of",
#       "Danielle Darrieux"s mother tongue is",
#       "Danielle Darrieux was born in",
#       "Danielle Darrieux"s mother tongue is",
#       "Danielle Darrieux"s mother tongue is",
#       "Danielle Darrieux was born in",
#       "Where Danielle Darrieux is from, people speak the language of",
#       "Danielle Darrieux was born in",
#       "Danielle Darrieux was born in"
#     ]
#   },
