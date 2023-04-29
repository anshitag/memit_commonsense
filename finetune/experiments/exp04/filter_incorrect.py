import argparse, json, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default='gpt2-large')
    parser.add_argument('-d', '--dataset', type=str, default='pep3k')
    parser.add_argument('-dt', '--datatype', type=str, default='normal')
    parser.add_argument('-o', '--output', type=str, default='incorrect-data/{model}_{dataset}_{datatype}_valid.json')

    args = parser.parse_args()
    output = args.output.format(model=args.model, dataset=args.dataset, datatype=args.datatype)

    with open(args.input, 'r') as f:
        data = json.load(f)

    out_data = []
    label_dic = { ' False': 0, ' True': 1 }
    pos, neg = 0, 0
    for d in data:
        if d['label'] != d['predicted_label']:
            if d['label'] == ' True':
                pos += 1
            else:
                neg += 1
            out_data.append({
                'prompt': d['prompt'],
                'label': label_dic[d['label']]
            })

    print('no of positive labels:', pos)
    print('no of negative labels:', neg)
    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output, 'w+') as f:
        json.dump(out_data, f, indent=4) 