import argparse
import json
from tqdm import tqdm
import os
import random
import unicodedata

def read_json(input_file):
    data = []
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def normalize_text(text):
    return unicodedata.normalize("NFKD", text)

def update_document(metrics, prediction, gold):
    cur_sp_pred = set(prediction)

    gold_sp_pred = set(gold)

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['doc_em'] += em
    metrics['doc_f1'] += f1
    metrics['doc_prec'] += prec
    metrics['doc_recall'] += recall
    return em, prec, recall, f1

def convert2para(data_list, max_para_num=15, is_training=False, converted_keys='context', sub_key=''):
    print('max para num is: {}, is training: {} convert keys is: {}, convert sub key is :{}'.format(max_para_num,
                                                                            is_training, converted_keys, sub_key))
    para_examples = []
    negtive = 0
    positive = 0
    converted_keys = converted_keys.split('_')
    metrics = {'doc_em': 0., 'doc_f1': 0., 'doc_prec': 0., 'doc_recall': 0.}
    pred_num = 0
    for example in tqdm(data_list):
        pred_num += 1
        para_id = 0
        _id = example['_id']
        answer = example['answer']
        question = example['question']
        sf = example['supporting_facts']
        sf_dict = {x[0]:x[1] for x in sf}
        if is_training:
            assert 'context' in converted_keys
        context =[]
        if 'context' in converted_keys:
            context = example['context']
        type = example['type']
        level = example['level']

        answers = [{'answer_text': answer, 'answer_start': -1}]
        para_num = 0
        context_merged = [x for x in context if x[0] in sf_dict]
        #only choose positive from context

        retrievals = []
        for converted_key in converted_keys:
            if converted_key == 'context':
                continue
            retrievals.extend(example[converted_key])
        if sub_key:
            retrievals = [x for x in retrievals if x[0] == sub_key]

        retrievals_dict = {x[2]:x for x in retrievals}
        retrievals = [x for _, x in retrievals_dict.items()]
        retrievals = sorted(retrievals, key=lambda x:x[1], reverse=True)
        for para in retrievals:
            context_merged.append([para[2], para[3]])
        context_merged_dict = {x[0]:x for x in context_merged}
        context_merged = [x for _, x in context_merged_dict.items()]
        context_merged_bk = context_merged[:]
        if is_training:
            if max_para_num > 4:
                context_merged = context_merged[:max_para_num-1]
                if context_merged_bk[max_para_num-1:]:
                    context_merged.append(random.choice(context_merged_bk[max_para_num-1:]))
            else:
                context_merged = context_merged[:max_para_num - 1]
        else:
            context_merged = context_merged[:max_para_num]
        example['selected'] = context_merged
        preds = [x[0] for x in context_merged]
        golds = [x[0] for x in sf]
        em, prec, recall, f1 = update_document(metrics, preds, golds)
        for para in context_merged:
            title = para[0]
            sentences = para[1]
            sentence_starts = []
            sentence_start = 0
            para_text = ''
            for sentence in sentences:
                sentence_starts.append(sentence_start)
                para_text += sentence
                sentence_start += len(sentence)
            sentence_ends = sentence_starts[1:] + [sentence_start]
            sentence_indices = [[x, y] for x, y in zip(sentence_starts, sentence_ends)]
            for sentence_index in range(len(sentences)):
                sent_start, sent_end = sentence_indices[sentence_index]
                assert para_text[sent_start:sent_end] == sentences[sentence_index]

            label = "0"
            if title in sf_dict:
                label = "1"
            para_id += 1
            para_example = {'id': title, 'squad_id': _id, 'question': question, 'document': para_text,
                            'squad_answers': answers, 'label': label, 'sentence_indices': sentence_indices}
            if label == '1':
                positive += 1
            else:
                negtive += 1
            para_num += 1
            para_examples.append(para_example)
    print('negtive num is {}, positive num is {}, rate is {}'.format(negtive, positive, negtive / positive))
    for key, value in metrics.items():
        metrics[key] /= pred_num
    print('pred num is {}, lables num is {}'.format(pred_num, len(data_list)))
    print('results is {}'.format(metrics))
    return para_examples

def write_line_json(examples, output_file):
    print('write {} examples to {}'.format(len(examples), output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write('\n')



def write_json(data, file):
    print('write squad format data to {}'.format(file))
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def convert2squad(data, option='gold'):
    yes_nums = 0
    no_nums = 0
    str_nums = 0
    squad_data = {'data': [], 'version': 'convert from hotpot qa'}
    for example in tqdm(data, total=len(data), desc='convert hotpot qa to squad format'):

        _id = example['_id']
        answer = example['answer']
        if answer == 'yes':
            yes_nums += 1
        elif answer == 'no':
            no_nums += 1
        else:
            str_nums += 1
        question = example['question']
        type = example.get('type', '')
        level = example.get('level', '')
        sf = example.get('supporting_facts', [])
        sf_dict = {x[0]: x[1] for x in sf}
        answer_text = ''
        answer_start = -1
        is_impossible = True
        answers = []
        sentence_labels = []
        sentence_titles = []

        context = ''
        sentences = []
        if option == 'gold':

            for index in range(len(example['context'])):
                example['context'][index][-1][-1] = example['context'][index][-1][-1] + ' '
            context = example['context']
            context_dict = {x[0]: x[1] for x in context}

            for one_sf in sf:
                title = one_sf[0]
                sentence_num = one_sf[1]
                para_list = context_dict[title]
                for sentence_index in range(len(para_list)):
                    if sentence_index == sentence_num:
                        sentence_labels.append("1")
                    else:
                        sentence_labels.append("0")
                    sentences.append(para_list[sentence_index])
                    sentence_titles.append(title)

        elif option == 'selected':
            for index in range(len(example['selected'])):
                example['selected'][index][-1][-1] = example['selected'][index][-1][-1] + ' '
            context = example['selected']
            context_dict = {x[0]: x[1] for x in context}
            for title, para_list in context_dict.items():
                sentence_num = sf_dict.get(title, -1)
                for sentence_index in range(len(para_list)):
                    if sentence_index == sentence_num:
                        sentence_labels.append("1")
                    else:
                        sentence_labels.append("0")
                    sentences.append(para_list[sentence_index])
                    sentence_titles.append(title)

        else:
            raise ValueError('wrong option of converting squad!')

        sentence_starts = []
        sentence_start = 0
        context = ''
        for sentence in sentences:
            sentence_starts.append(sentence_start)
            context += sentence
            sentence_start += len(sentence)
        sentence_ends = sentence_starts[1:] + [sentence_start]
        sentence_indices = [[x, y] for x, y in zip(sentence_starts, sentence_ends)]
        for sentence_index in range(len(sentences)):
            sent_start, sent_end = sentence_indices[sentence_index]
            assert context[sent_start:sent_end] == sentences[sentence_index]
        context = normalize_text(context)
        answer = normalize_text(answer)
        if answer in "yes no":
            answer_text = ''
            answer_start = -1
        else:
            answer_text = answer
            answer_start = context.find(answer)
            if answer_start == -1 and option == 'gold':
                print('wrong example, skipped!')
                continue
            if answer_start == -1:
                print('warning! cannot find answers!')
            is_impossible = False

        answers.append({'text': answer_text, 'answer_start': answer_start,
                            'yes_no': True if answer in "yes no" else False, 'type': type, 'level': level,
                            'sp': sf,
                            'sentences': sentences, 'sentence_indices': sentence_indices,
                            "sentence_labels": sentence_labels, 'sentence_titles': sentence_titles})

                    # print(short_answers)
        qas = [{'question': example['question'], 'id': example['_id'], "answers": answers,
                'is_impossible': is_impossible}]
        paragraph = {'context': context,
                     'qas': qas,
                     }
        paragraphs = [paragraph]
        squad_data['data'].append({'title': 'title', 'paragraphs': paragraphs})
    print('yes nums is {}, no nums is {}, text answer nums is {}'.format(yes_nums, no_nums, str_nums))
    return squad_data


def convert2tri(data, max_negtive_num):
    tri_examples = []
    for example in tqdm(data, total=len(data), desc='convert hotpot qa to tri'):
        _id = example['_id']
        answer = example['answer']
        question = example['question']
        sf = example['supporting_facts']
        sf_dict = {x[0]:x[1] for x in sf}
        for index in range(len(example['context'])):
            example['context'][index][-1][-1] = example['context'][index][-1][-1] + ' '
        context = example['context']
        type = example['type']
        level = example['level']
        retrievals = example['retrieval']
        answers = [{'answer_text': answer, 'answer_start': -1}]
        see_para = {}

        context_merged = [x for x in context]
        for para in retrievals:
            context_merged.append([para[2], para[3]])
        labels = []
        paras  = []
        negtive_num = 0
        for para in context_merged:
            if negtive_num > max_negtive_num:
                break
            title = para[0]
            para_text = "".join(para[1])
            label = "0"
            if title in sf_dict:
                label = "1"


            if title not in see_para:
                see_para[title] = 1
                if label == '0':
                    negtive_num += 1
                labels.append(label)
                paras.append(para_text)
        paras = paras[:max_negtive_num + 2]
        # assert len(paras) == 10
        tri_examples.append({'question': example['question'], 'paras': paras, '_id': example['_id'], 'labels': labels})
    return tri_examples



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotpot_file', default='', help='official label and retrieved paras')
    parser.add_argument('--converted_keys', default='retrieval', type=str,help='the key to be converted')
    parser.add_argument('--max_para_num', default=10, type=int)
    parser.add_argument('--output_dir', default='data', type=str)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--sub_key', default='', type=str)

    parser.add_argument('--squad_key', default='gold', type=str)
    args = parser.parse_args()
    data = read_json(args.hotpot_file)

    dir_name = os.path.join(os.path.dirname(os.path.abspath(args.hotpot_file)), args.output_dir + '_' + str(args.max_para_num))
    if not os.path.exists(dir_name):
        print('dir does not exist! mkdir {}'.format(dir_name))
        os.mkdir(dir_name)
    else:
        print('{} exist, overwrite files!'.format(dir_name))
    file_name = args.hotpot_file.split('/')[-1]
    is_training = False
    if 'train' in args.hotpot_file:
        print('is training data')
        is_training = True
    else:
        print('=' * 10 + 'dev data' + '=' * 10)
    chunk_size = len(data) // args.split
    assert args.split > 0
    if not is_training:
        args.split = 1

    data_lists = [data[chunk_size * chunk_index: chunk_size * (chunk_index + 1)] for chunk_index in range(args.split)]
    for data_num, data_list in enumerate(data_lists):
        dir_name_sub = dir_name
        if len(data_lists) > 1:
            dir_name_sub = os.path.join(dir_name, 'chunk{}'.format(data_num))
            if not os.path.exists(dir_name_sub):
                os.mkdir(dir_name_sub)
        print('convert chunk {} of {}'.format(data_num + 1, len(data_lists)))
        para_exampels = convert2para(data_list, max_para_num=args.max_para_num, is_training=is_training,
                                     converted_keys=args.converted_keys, sub_key=args.sub_key)
        write_line_json(para_exampels, dir_name_sub + "/{}.para.part_{}_of_{}".format(file_name, data_num, args.split))
        #
        option = args.squad_key
        squad_data = convert2squad(data_list, option=option)
        squad_file = dir_name_sub + "/{}.squad.part_{}_of_{}.{}".format(file_name, data_num, args.split, args.squad_key)
        write_json(squad_data, squad_file)

        tri_file = dir_name_sub + "/{}.para.tri.part_{}_of_{}".format(file_name, data_num, args.split)
        tri_examples = convert2tri(data_list, max_negtive_num=args.max_para_num - 2)
        write_line_json(tri_examples, tri_file)



