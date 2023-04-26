import argparse, os, json


def load_files(xl_file_path, large_file_path):
    
    with open(xl_file_path, 'r') as xl_file:
        xl = json.load(xl_file)

    with open(large_file_path, 'r') as l_file:
        l = json.load(l_file)
        
    return xl, l


def incorrect(xl, l):
    
    if not len(l) == len(xl):
        print(len(l), len(xl))
    assert len(l) == len(xl)

    xl_count = 0
    l_count = 0
    union_incorrect = []
    intersection_incorrect = []

    for i in range(len(xl)):
        xl_wrong = True if xl[i]['label'] != xl[i]['predicted_label'] else False
        xl_count += 1 if xl_wrong else 0
        
        l_wrong = True if l[i]['label'] != l[i]['predicted_label'] else False
        l_count += 1 if l_wrong else 0

        if xl_wrong or l_wrong:
            union_incorrect.append({
                'id':i,
                'prompt':xl[i]['prompt'],
                'gpt2-xl_predicted_wrong':xl_wrong,
                'gpt2-l_predicted_wrong':l_wrong,
                'label': xl[i]['label']})
        
        if xl_wrong and l_wrong:
            intersection_incorrect.append({
                'id':i,
                'prompt':xl[i]['prompt'],
                'gpt2-xl_predicted_wrong':xl_wrong,
                'gpt2-l_predicted_wrong':l_wrong,
                'label': xl[i]['label']})
            
    print(f'\nNumber of incorrect predictions by gpt2-xl: {xl_count}\nNumber of incorrect predictions by gpt2-large: {l_count}\nIntersection of incorrect predictions: {len(intersection_incorrect)}\nUnion of incorrect predictions: {len(union_incorrect)}')
    return union_incorrect, intersection_incorrect


def dump_file(union_incorrect, union_output_file_path, intersection_incorrect, intersection_output_file_path):
    with open(union_output_file_path, 'w') as union_output_file:
        json.dump(union_incorrect, union_output_file, indent=4)

    with open(intersection_output_file_path, 'w') as intersection_output_file:
        json.dump(intersection_incorrect, intersection_output_file, indent=4)
        
    print(f'\nUnion output file generated at: {union_output_file_path}')
    print(f'Intersection output file generated at: {intersection_output_file_path}\n')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-xl', '--xl_file_path', type=str, default='result/outputs/gpt2-xl_pep3k_normal.json')
    parser.add_argument('-l', '--large_file_path', type=str, default='result/outputs/gpt2-large_pep3k_normal.json')
    parser.add_argument('-uo', '--union_output_file_path', type=str, default='result/incorrect_predictions/pep3k_normal_union_incorrect.json')
    parser.add_argument('-io', '--intersection_output_file_path', type=str, default='result/incorrect_predictions/pep3k_normal_intersection_incorrect.json')
    
    args = parser.parse_args()
    
    xl_file_path, large_file_path = args.xl_file_path, args.large_file_path
    union_output_file_path,  intersection_output_file_path = args.union_output_file_path, args.intersection_output_file_path
    
    xl, l = load_files(xl_file_path, large_file_path)
    
    union_incorrect, intersection_incorrect = incorrect(xl, l)
    
    dump_file(union_incorrect, union_output_file_path, intersection_incorrect, intersection_output_file_path)