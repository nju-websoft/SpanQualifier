# Script for MultiSpanQA evaluation
import os
import re
import json
import string
import difflib
import warnings
import numpy as np
from collections import Counter


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def eval_dicts(gold_dict, pred_dict, no_answer):
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]['answers']
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        if no_answer:
            avna += compute_avna(prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    if no_answer:
        eval_dict['AvNA'] = 100. * avna / total

    return eval_dict

def compute_avna(prediction, ground_truths):
    """Compute answer vs. no-answer accuracy."""
    return float(bool(prediction) == bool(ground_truths))

def get_entities(label, token):
    def _validate_chunk(chunk):
        if chunk in ['O', 'B', 'I']:
            return
        else:
            warnings.warn('{} seems not to be IOB tag.'.format(chunk))
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i,s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(('O', -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ['O']]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ['O']]

    for i, chunk in enumerate(label + ['O']):
        _validate_chunk(chunk)
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((' '.join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag

    return chunks


def end_of_chunk(prev_tag, tag):
    chunk_end = False
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    return chunk_end

def start_of_chunk(prev_tag, tag):
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    return chunk_start


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_lcsubstr(s1, s2):
    list1 = s1.split(' ')
    list2 = s2.split(' ')
    s1 = list1
    s2 = list2
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax, s1, s2

def compute_scores(golds, preds, eval_type='em',average='micro'):


    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for k in list(golds.keys()):
        # print('k:',k)
        # print('v:',golds[k])
        gold = golds[k]
        pred = preds[k]
        # print('pred:', pred)
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type == 'em':
            # if len(gold) == 0 and len(pred) == 0:
            #     # print(len(gold.intersection(pred)))
            #     nb_correct += 1
            # else:
            #     nb_correct += len(gold.intersection(pred))
            if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
                nb_correct += 1
            nb_correct += len(gold.intersection(pred))
        else:
            p_score, r_score = count_overlap(gold, pred)
            nb_correct_p += p_score
            nb_correct_r += r_score
        # if (len(gold.intersection(pred)) / max(len(gold), 1)) != 1.0:
        #     print(k, len(gold.intersection(pred))/max(len(gold), 1))
        #     print('gold:', gold)
        #     print('pred:', pred)

    if eval_type == 'em':
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_gold if nb_gold > 0 else 0
    else:
        p = nb_correct_p / nb_pred if nb_pred > 0 else 0
        r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p,r,f


def count_overlap(gold, pred):
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1,1
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0,0
    p_scores = np.zeros((len(gold),len(pred)))
    r_scores = np.zeros((len(gold),len(pred)))
    for i,s1 in enumerate(gold):
        for j, s2 in enumerate(pred):

            s = difflib.SequenceMatcher(None, s1, s2)
            _,_,longest = s.find_longest_match(0, len(s1), 0, len(s2))
            p_scores[i][j] = longest / len(s2) if longest > 0 else 0
            r_scores[i][j] = longest / len(s1) if longest > 0 else 0

            # longest_str, longest, s1_list, s2_list = find_lcsubstr(s1, s2)
            # p_scores[i][j] = longest/len(s2_list) if longest>0 else 0
            # r_scores[i][j] = longest/len(s1_list) if longest>0 else 0

    p_score = sum(np.max(p_scores,axis=0))
    r_score = sum(np.max(r_scores,axis=1))
    return p_score, r_score


def read_gold(gold_file):
    with open(gold_file, encoding='utf-8') as f:
        data = json.load(f)['data']
        golds = {}
        for piece in data:
            if 'label' not in piece:
                piece['label'] = ['O'] * len(piece['context'])
            spans = list(set(map(lambda x: x[0], get_entities(piece['label'], piece['context']))))
            golds[piece['id']] = spans
    return golds


def read_pred(pred_file):
    with open(pred_file, encoding='utf-8') as f:
        preds = json.load(f)
    return preds


def multi_span_evaluate_from_file(pred_file, gold_file):
    preds = read_pred(pred_file)
    golds = read_gold(gold_file)
    result = multi_span_evaluate(preds, golds)
    return result


def answer_number_acc(preds, golds):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()
    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
        # if '' in golds[k]:
        #     golds[k].remove('')

    for k,v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))
        # if '' in preds[k]:
        #     preds[k].remove('')
    count = 0
    for k in golds.keys():
        if len(golds[k]) == len(preds[k]):
            count += 1
    return round(count / len(golds), 4) * 100




def multi_span_evaluate(preds, golds, brief=False):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()
    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
        # if '' in golds[k]:
        #     golds[k].remove('')

    for k,v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))
        # if '' in preds[k]:
        #     preds[k].remove('')
    # Evaluate
    em_p, em_r, em_f = compute_scores(golds, preds, eval_type='em')
    overlap_p, overlap_r, overlap_f = compute_scores(golds, preds, eval_type='overlap')
    if brief:
        result = {
                  'em_f1': 100 * round(em_f, 4),
                  'overlap_f1': 100 * round(overlap_f, 4)}
        return result
    else:
        result = {'em_precision': 100 * round(em_p, 4),
                  'em_recall': 100 * round(em_r, 4),
                  'em_f1': 100 * round(em_f, 4),
                  'overlap_precision': 100 * round(overlap_p, 4),
                  'overlap_recall': 100 * round(overlap_r, 4),
                  'overlap_f1': 100 * round(overlap_f, 4)}
        return result


# ------------ START: This part is for nbest predictions with confidence ---------- #

def eval_with_nbest_preds(nbest_file, gold_file):
    """ To use this part, check nbest output format of huggingface qa script """
    best_threshold,_ = find_best_threshold(nbest_file, gold_file)
    nbest_preds = read_nbest_pred(nbest_file)
    golds = read_gold(gold_file)
    preds = apply_threshold_nbest(best_threshold, nbest_preds)
    return multi_span_evaluate(preds, golds)


def check_overlap(offsets1, offsets2):
    if (offsets1[0]<=offsets2[0] and offsets1[1]>=offsets2[0]) or\
       (offsets1[0]>=offsets2[0] and offsets1[0]<=offsets2[1]):
        return True
    return False

def remove_overlapped_pred(pred):
    new_pred = [pred[0]]
    for p in pred[1:]:
        no_overlap = True
        for g in new_pred:
            if check_overlap(p['offsets'],g['offsets']):
                no_overlap = False
        if no_overlap:
            new_pred.append(p)
    return new_pred

def read_nbest_pred(nbest_pred_file):
    with open(nbest_pred_file) as f:
        nbest_pred = json.load(f)
    # Remove overlapped pred and normalize the answer text
    for k,v in nbest_pred.items():
        new_v = remove_overlapped_pred(v)
        for vv in new_v:
            vv['text'] = normalize_answer(vv['text'])
        nbest_pred[k] = new_v
    return nbest_pred

def apply_threshold_nbest(threshold, nbest_preds):
    preds = {}
    for k,v in nbest_preds.items():
        other_pred = filter(lambda x: x['probability']>= threshold, nbest_preds[k][1:]) # other preds except the first one
        if nbest_preds[k][0]['text'] != '': # only apply to the has_answer examples
            preds[k] = list(set([nbest_preds[k][0]['text']] + list(map(lambda x: x['text'], other_pred))))
        else:
            preds[k] = ['']
    return preds

def threshold2f1(threshold, golds, nbest_preds):
    preds = apply_threshold_nbest(threshold, nbest_preds)
    _,_,f1 = compute_scores(golds, preds, eval_type='em')
    return f1

def find_best_threshold(nbest_dev_file, gold_dev_file):
    golds = read_gold(gold_dev_file)
    nbest_preds = read_nbest_pred(nbest_dev_file)
    probs = list(map(lambda x:x[0]['probability'], nbest_preds.values()))
    sorted_probs = sorted(probs, reverse=True)
    # search probs in prob list and find the best threshold
    best_threshold = 0.5
    best_f1 = threshold2f1(0.5, golds, nbest_preds)
    for prob in sorted_probs:
        if prob > 0.5:
            continue
        cur_f1 = threshold2f1(prob, golds, nbest_preds)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = prob
    return best_threshold, best_f1
# ------------ END: This part is for nbest predictions with confidence ---------- #

def read_gold_quoref(gold_file):
    gold_answers = {}
    with open(gold_file, encoding='utf-8') as f:
        dataset = json.load(f)['data']
    for sample in dataset:
        paragraphs = sample['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for qa in qas:
                id = qa['id']
                answers = qa['answers']
                answers = [item['text'] for item in answers]
                gold_answers[id] = answers
    return gold_answers

