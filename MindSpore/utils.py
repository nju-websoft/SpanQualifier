#coding=utf-8
import json
import os
import numpy as np
import random
import re
from eval_script import get_entities
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def read_dataset(path: str):
    if path.endswith('jsonl'):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    if 'data' in dataset:
        dataset = dataset['data']
    return dataset

def read_dataset_wo_results(path, result_path=None):
    dataset = read_dataset(path)
    prediction = {}
    reference = []
    if result_path is not None and os.path.exists(result_path):
        prediction = read_dataset(result_path)
        reference = [sample for sample in dataset if sample['id'] in prediction.keys()]
        dataset = [sample for sample in dataset if sample['id'] not in prediction.keys()]

    return dataset, reference, prediction

def append_result(fold_path, file_name, sample):
    os.makedirs(fold_path, exist_ok=True)
    with open(fold_path + '/' + file_name, 'a', encoding='utf-8') as f:
        if 'ctxs' in sample:
            del sample['ctxs']
        line = json.dumps(sample)
        f.write(line + '\n')

def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")

# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0



def read_quoref(path):
    dataset = read_dataset(path)
    dataset_new = []
    for sample in dataset:
        paragraphs = sample['paragraphs']
        for p_samples in paragraphs:
            context = p_samples['context']
            qas = p_samples['qas']
            for qa_sample in qas:
                question = qa_sample['question']
                id = qa_sample['id']
                answers = qa_sample['answers']
                answers = sorted(answers, key=lambda x: x['answer_start'])
                answers_idx = []
                answers_text = []
                for answer_item in answers:
                    text = answer_item['text']
                    answer_start = answer_item['answer_start']
                    answer_end = answer_start + len(text)
                    answers_idx.append([answer_start, answer_end])
                    assert context[answer_start: answer_end] == text
                    answers_text.append(text)
                dataset_new.append({
                    'id': id,
                    'question': question,
                    'context': context,
                    'answers': answers_text,
                    'answers_idx': answers_idx
                })
    return dataset_new



def read_msqa(path):
    dataset = read_dataset(path)
    dataset_new = []
    for sample in dataset:
        id = sample['id']
        question = sample['question']
        context = sample['context']
        label = sample['label']
        answers_w_idx = get_entities(label, context)
        answers_w_idx = sorted(answers_w_idx, key=lambda x: x[1])
        answers = [item[0] for item in answers_w_idx]
        context_char = ""
        context_char_idx_beg, context_char_idx_end = [], []
        beg_idx = 0
        for word in context:
            context_char_idx_beg.append(beg_idx)
            context_char_idx_end.append(beg_idx + len(word))
            beg_idx += len(word) + 1
            context_char += word + ' '
        context_char = context_char.strip()

        answers_idx_char = []
        for ans, beg_idx, end_idx in answers_w_idx:
            # if context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]] != ans:
            #     print(context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]])
            #     print(ans)
            assert context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]] == ans
            answers_idx_char.append([
                context_char_idx_beg[beg_idx],
                context_char_idx_end[end_idx],
            ])
        dataset_new.append({
            'id': id,
            'question': ' '.join(question),
            'context': context_char,
            'answers': answers,
            'answers_idx': answers_idx_char
        })
    return dataset_new


def save_model(output_model_file, model, model_name):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += model_name
    mindspore.save_checkpoint(model, output_model_file)


def split_sequence(bert_output_word_q, background_range, useSep=True, set_max_len=None):
    useSep_offset = 0
    if useSep:
        useSep_offset = 1
    batch, seq_len, dim = bert_output_word_q.shape
    max_len = 0
    for b_range in background_range:
        q_beg = b_range[0]
        c_end = b_range[1]
        question_len = c_end - q_beg + 1
        if question_len > max_len:
            max_len = question_len
    if useSep:
        max_len += 1
    if set_max_len is not None:
        if set_max_len > max_len:
            max_len = set_max_len
    bert_output_word_q = bert_output_word_q.reshape(-1, dim)
    index_select = []
    offset = 0
    step_size = seq_len
    mask = []
    tmp = 0
    for index in range(batch):
        b_range = background_range[index]
        q_beg = b_range[0]
        c_end = b_range[1] + useSep_offset
        if c_end >= seq_len:
            print('c_end < seq_len:', c_end, seq_len)

        assert c_end < seq_len
        index_select_item = []
        mask_item = []
        for i in range(q_beg, c_end + 1):
            index_select_item.append(offset * step_size + i + 1)
            if offset * step_size + i + 1 > tmp:
                tmp = offset * step_size + i + 1
            mask_item.append(1)
        while len(index_select_item) < max_len:
            index_select_item.append(0)
            mask_item.append(0)
        index_select += index_select_item
        mask.append(mask_item)
        offset += 1

    index_select = Tensor(index_select, mindspore.int32)
    mask = Tensor(mask, mindspore.int32)

    sequence_output = bert_output_word_q.reshape(-1, dim)
    sequence_output = mindspore.ops.cat((sequence_output.new_zeros((1, dim)), sequence_output), 0)
    sequence_new = sequence_output.index_select(0, index_select)
    sequence_new = sequence_new.view(batch, max_len, dim)
    return sequence_new, mask