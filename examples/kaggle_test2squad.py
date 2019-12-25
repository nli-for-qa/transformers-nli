import json
import argparse
def read_line_json(input_file):
    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip('\n')))
    return examples

def convert_nq2squad(examples):
    squad_data = {'data': [], 'version': 'converted from nq kaggle'}
    for example in examples:
        context = example['document_text']
        nq_context_map = [i for i in range(len(context.split()))]
        nq_long_candidates = [[candidate['start_token'], candidate['end_token']] for candidate in example['long_answer_candidates']]
        long_answer_candidates = example['long_answer_candidates']
        answers = []
        if example.get('annotations', None):
            annotations = example['annotations']
            for annotation in annotations:
                short_answers = annotation['short_answers']
                if short_answers:
                    for short_answer in short_answers:
                        answers.append({'text': " ".join(context.split()[short_answer['start_token']:short_answer['end_token']]), 'answer_start': -1})
                    # print(short_answers)
        qas = [{'question': example['question_text'], 'id': example['example_id'], "answers": answers}]
        paragraph = {'context': context,
                     'nq_context_map': nq_context_map,
                     'nq_long_candidates': nq_long_candidates,
                     'qas': qas}
        paragraphs = [paragraph]
        squad_data['data'].append({'title': 'title', 'paragraphs': paragraphs})
    return squad_data

def write_json(output_file, data):
    print('write squad data to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(data, f)

def convert_kaggle2squad(input_file, output_file):
    nq_examples = read_line_json(input_file)
    squad_data = convert_nq2squad(nq_examples)
    write_json(output_file, squad_data)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle_test_file', help='kaggle test file nq-test.jsonl')
    parser.add_argument('--output_file', help='output file name')
    args = parser.parse_args()
    # args.output_file = args.kaggle_test_file + '.squad'
    convert_kaggle2squad(args.kaggle_test_file, args.output_file)
