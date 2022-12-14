#coding=utf-8
import json
import os
import torch
import numpy as np
import random
import re



def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

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

def split_sequence_multi_span(bert_output_word_q, batch_ranges):
    batch = bert_output_word_q.size(0)
    seq_len = bert_output_word_q.size(1)
    dim = bert_output_word_q.size(2)
    max_len = 0
    for ranges in batch_ranges:
        for beg, end in ranges:
            question_len = end - beg + 1
            if question_len > max_len:
                max_len = question_len
    if max_len == 0:
        return None, None
    bert_output_word_q = bert_output_word_q.reshape(-1, dim)
    index_select = []
    offset = 0
    step_size = seq_len
    mask = []
    tmp = 0
    for index in range(batch):
        ranges = batch_ranges[index]
        for beg, end in ranges:
            assert end < seq_len
            index_select_item = []
            mask_item = []
            for i in range(beg, end + 1):
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

    index_select = torch.tensor(index_select, dtype=torch.long)
    index_select = index_select.cuda()
    mask = torch.tensor(mask, dtype=torch.long)
    mask = mask.cuda()
    sequence_output = bert_output_word_q.reshape(-1, dim)
    sequence_output = torch.cat([sequence_output.new_zeros((1, dim), dtype=torch.float), sequence_output],
                                dim=0)
    sequence_new = sequence_output.index_select(0, index_select)
    sequence_new = sequence_new.view(-1, max_len, dim)
    return sequence_new, mask

def split_sequence(bert_output_word_q, background_range, useSep=True, set_max_len=None):
    useSep_offset = 0
    if useSep:
        useSep_offset = 1
    batch = bert_output_word_q.size(0)
    seq_len = bert_output_word_q.size(1)
    dim = bert_output_word_q.size(2)
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

    index_select = torch.tensor(index_select, dtype=torch.long)
    index_select = index_select.cuda()
    mask = torch.tensor(mask, dtype=torch.long)
    mask = mask.cuda()
    sequence_output = bert_output_word_q.reshape(-1, dim)
    sequence_output = torch.cat([sequence_output.new_zeros((1, dim), dtype=torch.float), sequence_output],
                                dim=0)
    sequence_new = sequence_output.index_select(0, index_select)
    sequence_new = sequence_new.view(batch, max_len, dim)
    return sequence_new, mask


def read_dataset(path):
    f = open(path, 'r', encoding='utf-8')
    dataset = json.load(f)
    dataset = dataset['data']
    return dataset

def save_model(output_model_file, model, optimizer):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += 'pytorch_model.bin'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model_file, _use_new_zipfile_serialization=False)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
