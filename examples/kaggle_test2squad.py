import json
import argparse
import re
from tqdm import tqdm
import numpy as np

def read_line_json(input_file):
    print('read kaggle test from input file {}'.format(input_file))
    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip('\n')))
    return examples

def convert_nq2squad(examples):
    print('convert examples of kaggle to squad fromat')
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
                     'nq_long_candidates_orig': example['long_answer_candidates'],
                     'qas': qas}
        paragraphs = [paragraph]
        squad_data['data'].append({'title': 'title', 'paragraphs': paragraphs})
    return squad_data

def write_json(output_file, data):
    print('write squad data to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(data, f)

def clean_data(context, nq_context_map):
    doc_tokens = context.split()
    html_regex = re.compile(r'<.*>')
    tokens = []
    maps = []
    for token, map_id in zip(doc_tokens, nq_context_map):
        if html_regex.search(token):
            continue
        else:
            tokens.append(token)
            maps.append(map_id)
    return " ".join(tokens), maps

def convert_nq2para(squad_data, clean=False, chunk_len=256):
    print('convert nq sqaud format to paras')
    data = squad_data['data']
    para_examples = []
    para_index = 0
    for paragraphs in tqdm(data, desc='convert squad format to paras'):
        for paragraph in paragraphs['paragraphs']:
            context = paragraph['context']
            nq_context_map = paragraph['nq_context_map']
            nq_long_candidates = paragraph['nq_long_candidates']
            nq_long_candidates_orig = paragraph['nq_long_candidates_orig']
            top_paras = [[candidate['start_token'], candidate['end_token']]
                         for candidate in nq_long_candidates_orig if candidate['top_level']]

            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                _id = qa['id']
                answers = qa['answers']
                if clean:
                    context, nq_context_map = clean_data(context, nq_context_map)
                doc_tokens = context.split()
                for para in top_paras:
                    para_tokens = doc_tokens[para[0]:para[1]]
                    para_map = nq_context_map[para[0]:para[1]]
                    assert len(para_tokens) == len(para_map)
                    para_context = " ".join(para_tokens)
                    para_example = {'id': para_index, 'squad_id': _id, 'question': question, 'document': para_context,
                                    'squad_answers': answers, 'para_context_map': para_map,
                                    'nq_long_candidates': nq_long_candidates, 'label': '0'}
                    para_examples.append(para_example)
                    para_index += 1
    return para_examples

def write_para(para_examples, output_file_para):
    with open(output_file_para, 'w') as f:
        for example in para_examples:
            f.write(json.dumps(example))
            f.write('\n')




def convert_kaggle2squad(input_file, output_file, clean_data=False):
    nq_examples = read_line_json(input_file)
    squad_data = convert_nq2squad(nq_examples)
    para_examples = convert_nq2para(squad_data, clean_data)
    output_file_para = output_file + '.para'
    write_para(para_examples, output_file_para)

    write_json(output_file, squad_data)

def combine_example_cls(para_file, eval_results):
    print('para_file is {}'.format(para_file))
    print('eval_results is {}'.format(eval_results))
    examples = []
    with open(para_file, 'r') as fin:
        for line in tqdm(fin):
            examples.append(json.loads(line.strip('\n')))

    with open(eval_results) as fin:
        eval_result = json.load(fin)
    preds = eval_result['preds']
    logits = eval_result['logits']
    labels = eval_result['labels']
    print('preds: ', len(preds), 'examples:', len(examples))
    assert len(preds) == len(examples)
    assert len(labels) == len(examples)
    for i, example in tqdm(enumerate(examples)):
        assert example['label'] == str(labels[i])
        examples[i]['cls_score'] = logits[i]
    print('write para_file with cls scores to {}'.format(para_file))
    with open(para_file, 'w', encoding='utf-8') as fout:
        for line in examples:
            fout.write(json.dumps(line))
            fout.write('\n')

def read_para_input(input_file):
    dataset = {}
    print('read para input from {}'.format(input_file))
    with open(input_file, 'r') as fin:
        for line in tqdm(fin, desc='read input'):
            line = json.loads(line.strip('\n'))
            if line['squad_id'] not in dataset:
                dataset[line['squad_id']] = {'question': line['question'],
                                       'contexts': [line['document']],
                                       'answers': line['squad_answers'],
                                        'para_context_maps': [line['para_context_map']],
                                        'nq_long_candidates': line['nq_long_candidates'],
                                       'id': line['squad_id'],
                                       'document_cls_scores': [line.get('cls_score', None)],
                                        'labels': [line.get('label', None)],
                                       }
            else:
                dataset[line['squad_id']]['contexts'].append(line['document'])
                dataset[line['squad_id']]['document_cls_scores'].append((line.get('cls_score', None)))
                dataset[line['squad_id']]['labels'].append(line.get('label', None))
                dataset[line['squad_id']]['para_context_maps'].append(line['para_context_map'])
    print('data set num with squad id as key is {}'.format(len(dataset)))
    return dataset

def select_top_n(para_dict, top_n=5, para_file=''):
    print('select top {} paras'.format(top_n))
    total_cnt = 0
    squad_qa = {'data': [], 'version': 'converted from cls'}
    for squad_id, squad_question_content in tqdm(para_dict.items(), desc='convert to qa format',
                                                 total=len(para_dict.keys())):
        total_cnt += 1
        article = {}
        top_n = top_n

        squad_question_content_positive_scores = [score[1] for score in
                                                      squad_question_content['document_cls_scores']]
        sorted_indices = np.argsort(squad_question_content_positive_scores)[::-1].tolist()
            # contexts = select_paras(squad_question_content['contexts'], squad_question_content['document_cls_scores'],
            #                         args.cls_top_n, options='cls')
        assert len(squad_question_content['labels']) == len(squad_question_content['document_cls_scores'])
        topn_indices = sorted_indices[:top_n]

        contexts = [squad_question_content['contexts'][index] for index in topn_indices]
        contexts = " ".join(' '.join(contexts).split())
        context_map = [squad_question_content['para_context_maps'][index] for index in topn_indices]
        context_map = [tmp for tmp_list in context_map for tmp in tmp_list]
        assert len(contexts.split()) == len(context_map)

        paragraph = {'context': contexts}
        paragraph['nq_long_candidates'] = squad_question_content['nq_long_candidates']
        paragraph['nq_context_map'] = context_map
        qa = {'question': squad_question_content['question'],
              'answers': squad_question_content['answers'],
              'id': squad_id}

        paragraph['qas'] = [qa]
        paragraphs = [paragraph]
        article['paragraphs'] = paragraphs
        squad_qa['data'].append(article)

    qa_file = '{}.qa.top_{}'.format(para_file, top_n)
    print('write_selected qa examples to {}'.format(qa_file))
    with open(qa_file, 'w') as f:
        json.dump(squad_qa, f)

def cls2qa(para_file, eval_results_file, top_n=5):

    combine_example_cls(para_file, eval_results_file)
    para_dict = read_para_input(para_file)
    select_top_n(para_dict, top_n=top_n, para_file=para_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert_option', default='convert2squad_paras', type=str)
    parser.add_argument('--kaggle_test_file', help='kaggle test file nq-test.jsonl')
    parser.add_argument('--output_file', help='output file name')
    parser.add_argument('--clean_data', action='store_true')

    parser.add_argument('--para_file', default='', type=str, help='paras in a file')
    parser.add_argument('--eval_results', default='', type=str, help='eval_results from cls')
    parser.add_argument('--top_n', default=5, type=int)
    args = parser.parse_args()
    # args.output_file = args.kaggle_test_file + '.squad'
    if args.convert_option == 'convert2squad_paras':
        convert_kaggle2squad(args.kaggle_test_file, args.output_file, args.clean_data)
    elif args.convert_option == 'combine_example_cls':

        cls2qa(args.para_file, args.eval_results, top_n=args.top_n)
