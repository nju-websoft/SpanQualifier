# coding=utf-8
from transformers import AutoTokenizer, BertTokenizerFast, AlbertTokenizerFast, DebertaTokenizerFast, AutoModel, get_linear_schedule_with_warmup
from tqdm import trange
import os
import random
import torch
from utils import save_dataset, read_msqa, read_quoref, set_seed, save_model, split_sequence
import json
import argparse
from torch import nn
import math
from collections import OrderedDict
from eval_script import multi_span_evaluate
import copy
import ast

device = torch.device("cuda:0")
class MLP(nn.Module):
    def __init__(self, dim0, dim1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(dim0, dim0)
        self.linear2 = nn.Linear(dim0, dim1)
        self.activate = nn.ReLU()

    def forward(self, input):
        input = self.linear1(input)
        input = self.activate(input)
        input = self.linear2(input)
        return input

class BoundaryEnumeration(nn.Module):
    def __init__(self, dim):
        super(BoundaryEnumeration, self).__init__()
        self.s_boundary_enum = MLP(dim, dim)
        self.e_boundary_enum = MLP(dim, dim)
    def forward(self, H_c):
        B_s = self.s_boundary_enum(H_c)
        B_e = self.e_boundary_enum(H_c)
        return B_s, B_e

class BoundaryRepresentation(nn.Module):
    def __init__(self, dim1, dim2, max_len, max_span_gap, vanilla=False):
        super(BoundaryRepresentation, self).__init__()
        self.boundary_enum = BoundaryEnumeration(dim1)
        self.vanilla = vanilla
        self.boundary_aggregation = BoundaryAggregation(dim1, dim2, max_len, max_span_gap)

    def forward(self, H_c, H_cls, masks):
        B_s, B_e = self.boundary_enum(H_c)
        G_s, G_e, qs_s, qs_e = None, None, None, None
        if self.vanilla is False :
            B_s, B_e, G_s, G_e, qs_s, qs_e = self.boundary_aggregation(B_s, B_e, H_cls, masks)
        return B_s, B_e, G_s, G_e, qs_s, qs_e

class SpanEnumeration(nn.Module):
    def __init__(self, dim1, dim2, max_len):
        super(SpanEnumeration, self).__init__()
        self.s_mapping = nn.Linear(dim1, dim2)
        self.e_mapping = nn.Linear(dim1, dim2)
        self.pos_embedding = nn.Embedding(max_len, dim2)
        self.layer_norm = nn.LayerNorm(dim2, eps=1e-12)
        pos_id = []
        for i in range(max_len):
            for j in range(max_len):
                pos_id.append(int(math.fabs(j - i)))
        self.pos_id = torch.tensor(pos_id, dtype=torch.long).to(device)
        self.dim2 = dim2
        self.max_len = max_len

    def forward(self, B_s, B_e):
        bs, seq_len, dim = B_s.size()
        pos_embedding = self.pos_embedding(self.pos_id).view(self.max_len, self.max_len, self.dim2)
        pos_embedding = pos_embedding[:seq_len, :seq_len, :]
        pos_embedding = pos_embedding.reshape(seq_len, seq_len, self.dim2)
        pos_embedding = pos_embedding.unsqueeze(dim=0).expand(bs, seq_len, seq_len, self.dim2)

        B_s = self.s_mapping(B_s)
        B_e = self.s_mapping(B_e)

        B_s_ex = B_s.unsqueeze(dim=2).expand([bs, seq_len, seq_len, self.dim2])
        B_e_ex = B_e.unsqueeze(dim=2).expand([bs, seq_len, seq_len, self.dim2])
        B_e_ex = torch.transpose(B_e_ex, dim0=1, dim1=2)
        N = B_s_ex + B_e_ex + pos_embedding
        M = self.layer_norm(N)
        return M


class SpanRepresentation(nn.Module):
    def __init__(self, dim1, dim2, max_len, vanilla=False):
        super(SpanRepresentation, self).__init__()
        self.span_enum = SpanEnumeration(dim1, dim2, max_len)
        self.span_interaction = SpanInteraction(dim2)
        self.vanilla = vanilla
        masks_triangle = []
        for i in range(args.max_len):
            for j in range(args.max_len):
                if i <= j and j - i <= max_span_gap:
                    masks_triangle.append(1)
                else:
                    masks_triangle.append(0)
        self.masks_triangle = torch.tensor(masks_triangle, dtype=torch.float).to(device).view(args.max_len,
                                                                                              args.max_len)

    def forward(self, B_s, B_e, masks):
        M = self.span_enum(B_s, B_e)

        bs, seq_len, dim = B_s.size()
        masks_c_ex = masks.unsqueeze(dim=1).expand(bs, seq_len, seq_len)
        masks_c_ex_t = torch.transpose(masks_c_ex, dim0=1, dim1=2)
        masks_c_ex = masks_c_ex * masks_c_ex_t
        masks_triangle = self.masks_triangle
        masks_triangle = masks_triangle[:seq_len, :seq_len]
        masks_triangle = masks_triangle.view(seq_len, seq_len)
        masks_triangle = masks_triangle.unsqueeze(dim=0).expand(bs, seq_len, seq_len)
        masks_triangle = masks_triangle.clone()
        masks_matrix = masks_c_ex * masks_triangle
        M = M * masks_matrix.unsqueeze(dim=3)
        if self.vanilla is False:
            M = self.span_interaction(M)
        return M

class SpanScoring(nn.Module):
    def __init__(self, dim1, dim2, max_span_gap):
        super(SpanScoring, self).__init__()
        self.mlp_scoring = MLP(dim2, 1)
        self.mlp_cls = MLP(dim1, 1)

        masks_triangle = []
        for i in range(args.max_len):
            for j in range(args.max_len):
                if i <= j and j - i <= max_span_gap:
                    masks_triangle.append(1)
                else:
                    masks_triangle.append(0)
        self.masks_triangle = torch.tensor(masks_triangle, dtype=torch.float).to(device).view(args.max_len,
                                                                                              args.max_len)
    def forward(self, M, H_cls, masks):

        S = self.mlp_scoring(M)
        qs = self.mlp_cls(H_cls)
        bs, seq_len, seq_len, dim = M.size()
        S = S.view(bs, seq_len, seq_len)
        masks_ex = masks.unsqueeze(dim=1).expand(bs, seq_len, seq_len)
        masks_ex_t = torch.transpose(masks_ex, dim0=1, dim1=2)
        masks_ex = masks_ex * masks_ex_t
        masks_triangle = self.masks_triangle
        masks_triangle = masks_triangle[:seq_len, :seq_len]
        masks_triangle = masks_triangle.view(seq_len, seq_len)
        masks_triangle = masks_triangle.unsqueeze(dim=0).expand(bs, seq_len, seq_len)
        masks_triangle = masks_triangle.clone()
        masks_matrix = masks_ex * masks_triangle
        S = S - 10000.0 * (1 - masks_matrix)
        return S, qs

class BoundaryAggregation(nn.Module):
    def __init__(self, dim1, dim2, max_len, max_span_gap):
        super(BoundaryAggregation, self).__init__()
        self.span_enum_s = SpanEnumeration(dim1, dim2, max_len)
        self.span_enum_e = SpanEnumeration(dim1, dim2, max_len)
        self.span_scoring_s = SpanScoring(dim1, dim2, max_span_gap)
        self.span_scoring_e = SpanScoring(dim1, dim2, max_span_gap)
        self.W2_s = nn.Linear(dim1, dim1)
        self.W2_e = nn.Linear(dim1, dim1)
        self.span_interaction_s = SpanInteraction(dim2)
        self.span_interaction_e = SpanInteraction(dim2)

    def forward(self, hB_s, hB_e, H_cls, masks):
        bs, seq_len, dim = hB_s.size()
        M_s = self.span_enum_s(hB_s, hB_e)
        M_s = self.span_interaction_s(M_s)
        G_s, qs_s = self.span_scoring_s(M_s, H_cls, masks)
        G_s_soft = torch.softmax(G_s, dim=-1)
        B_s = torch.matmul(G_s_soft, self.W2_s(hB_s))
        B_s = B_s.view(bs, seq_len, dim)
        M_e = self.span_enum_e(hB_s, hB_e)
        M_e = self.span_interaction_e(M_e)
        G_e, qs_e = self.span_scoring_e(M_e, H_cls, masks)
        G_e_soft = torch.softmax(torch.transpose(G_e, dim0=-2, dim1=-1), dim=-1)
        B_e = torch.matmul(G_e_soft, self.W2_e(hB_e))
        B_e = B_e.view(bs, seq_len, dim)

        return B_s, B_e, G_s, G_e, qs_s, qs_e

class SpanInteraction(nn.Module):
    def __init__(self, dim2):
        super(SpanInteraction, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim2,
                              out_channels=dim2,
                              kernel_size=(5, 5),
                              padding=(2, 2))
    def forward(self, hM):
        hM = hM.permute(0, 3, 1, 2)
        hM = self.conv(hM)
        M = hM.permute(0, 2, 3, 1)
        return M

class SpanQualifier(nn.Module):
    def __init__(self, model_path, max_span_gap, dim2, max_len, vanilla=False):
        super(SpanQualifier, self).__init__()
        self.token_representation = AutoModel.from_pretrained(model_path)
        dim1 = self.token_representation.config.hidden_size
        self.boundary_representation = BoundaryRepresentation(dim1, dim2, max_len, max_span_gap, vanilla)
        self.span_representation = SpanRepresentation(dim1, dim2, max_len, vanilla)
        self.span_scoring = SpanScoring(dim1, dim2, max_span_gap)
        self.vanilla = vanilla


    def forward(self, input_ids, type_ids, mask_ids, context_ranges, targets=None):

        outputs = self.token_representation(input_ids=input_ids,
                                            attention_mask=mask_ids,
                                            token_type_ids=type_ids,
                                            output_hidden_states=True,
                                            return_dict=True)
        sequence_output = outputs.hidden_states[-1]
        H_cls = sequence_output[:, 0, :].reshape(-1, sequence_output.size(-1))
        H_c, masks = split_sequence(sequence_output, context_ranges, useSep=False)
        B_s, B_e, G_s, G_e, qs_s, qs_e = self.boundary_representation(H_c, H_cls, masks)
        M = self.span_representation(B_s, B_e, masks)
        S, qs_ext = self.span_scoring(M, H_cls, masks)

        if targets is not None:
            loss, labels_batch = self.extract_loss(S, targets, qs_ext)
            if self.vanilla is False:
                loss += self.attention_loss_start(G_s, targets, qs_s)
                loss += self.attention_loss_end(G_e, targets, qs_e)
                loss = loss / 3
            return loss
        else:
            spans, spans_matrix = self.decoding_span_matrix(S, qs_ext)
            return spans

    def decoding_span_matrix(self, logits_matrix, threhold_p, spans_matrix_mask=None):
        bs, seq_len, seq_len = logits_matrix.size()

        if spans_matrix_mask is not None:
            logits_matrix = logits_matrix - 10000.0 * spans_matrix_mask
        logits_end = torch.softmax(logits_matrix, dim=2)
        _, idx_best_end = torch.max(logits_end, dim=2)
        idx_best_end = idx_best_end.cpu().tolist()
        threhold_p = threhold_p.view(bs)
        threhold_p = threhold_p.cpu().tolist()

        logits_beg = torch.softmax(logits_matrix, dim=1)

        _, idx_best_beg = torch.max(logits_beg, dim=1)
        idx_best_beg = idx_best_beg.cpu().tolist()

        logits_matrix = logits_matrix.cpu().tolist()
        spans = []
        spans_matrix = []
        for b_i, (matrix, t_p) in enumerate(zip(logits_matrix, threhold_p)):
            spans_item = []
            max_logit, max_i, max_j = -10000, 0, 0
            spans_matrix_item = [[0] * seq_len for i in range(seq_len)]
            for i, logits in enumerate(matrix):
                for j, logit in enumerate(logits):
                    if i <= j and idx_best_end[b_i][i] == j and idx_best_beg[b_i][j] == i:
                        if logit > t_p:
                            spans_item.append([i, j])
                            spans_matrix_item[i][j] = 1
                        if logit > max_logit:
                            max_logit = logit
                            max_i, max_j = i, j
            if len(spans_item) == 0 and force_answer:
                spans_item.append([max_i, max_j])
            spans.append(spans_item)
            spans_matrix.append(spans_matrix_item)
        spans_matrix = torch.tensor(spans_matrix, dtype=torch.float).to(device)
        return spans, spans_matrix


    def attention_loss_end(self, logits, span_targets, qs):
        bs, seq_len, seq_len = logits.size()
        qs = qs.view(-1)
        labels_batch = []
        has_answers = []
        loss = []
        global_has_answer = False
        for spans in span_targets:
            label_matrix = [[0] * seq_len for i in range(seq_len)]
            has_answer = 0
            for (beg, end) in spans:
                for j in range(beg, end + 1):
                    # label_matrix[beg][j] = 1
                    label_matrix[j][end] = 1
                has_answer = 1
                global_has_answer = True
            has_answers.append(has_answer)
            labels_batch.append(label_matrix)
        labels_batch = torch.tensor(labels_batch, dtype=torch.float).to(device)
        has_answers = torch.tensor(has_answers, dtype=torch.float).to(device)
        has_answers_idx = has_answers > 0
        neg_span = logits * (1 - labels_batch)
        neg_span_max, _ = torch.max(neg_span, dim=2)
        neg_span_max, _ = torch.max(neg_span_max, dim=1)
        loss_margin_neg = torch.clamp_min(1 - (qs - neg_span_max), 0)
        loss_margin_neg = torch.mean(loss_margin_neg, dim=0)
        loss.append(loss_margin_neg)

        if global_has_answer is False:
            return loss_margin_neg

        pos_span = 0 - logits * labels_batch
        pos_span = pos_span - 10000.0 * (1 - labels_batch)
        pos_span_min, _ = torch.max(pos_span, dim=2)
        pos_span_min, _ = torch.max(pos_span_min, dim=1)
        pos_span_min = 0 - pos_span_min
        loss_margin_pos = torch.clamp_min(1 - (pos_span_min - qs), 0)
        loss_margin_pos = torch.mean(loss_margin_pos[has_answers_idx])
        loss.append(loss_margin_pos)

        logits = logits.view(-1, seq_len * seq_len)
        labels_batch = labels_batch.view(-1, seq_len * seq_len)
        logits_soft = torch.softmax(logits, dim=1)
        loss_flat = torch.sum(logits_soft * labels_batch, dim=1)
        loss_flat = -torch.log(torch.clamp(loss_flat, 0.0001, 1))
        loss_flat = torch.mean(loss_flat[has_answers_idx], dim=0)
        loss.append(loss_flat)

        return sum(loss) / len(loss)

    def attention_loss_start(self, logits, span_targets, qs):
        bs, seq_len, seq_len = logits.size()
        qs = qs.view(-1)
        labels_batch = []
        has_answers = []
        loss = []
        global_has_answer = False
        for spans in span_targets:
            label_matrix = [[0] * seq_len for i in range(seq_len)]
            has_answer = 0
            for (beg, end) in spans:
                for j in range(beg, end+1):
                    label_matrix[beg][j] = 1
                    # label_matrix[j][end-1] = 1
                has_answer = 1
                global_has_answer = True
            has_answers.append(has_answer)
            labels_batch.append(label_matrix)
        labels_batch = torch.tensor(labels_batch, dtype=torch.float).to(device)
        has_answers = torch.tensor(has_answers, dtype=torch.float).to(device)
        has_answers_idx = has_answers > 0
        neg_span = logits * (1 - labels_batch)
        neg_span_max, _ = torch.max(neg_span, dim=2)
        neg_span_max, _ = torch.max(neg_span_max, dim=1)
        loss_margin_neg = torch.clamp_min(1 - (qs - neg_span_max), 0)
        loss_margin_neg = torch.mean(loss_margin_neg, dim=0)
        loss.append(loss_margin_neg)

        if global_has_answer is False:
            return loss_margin_neg

        pos_span = 0 - logits * labels_batch
        pos_span = pos_span - 10000.0 * (1 - labels_batch)
        pos_span_min, _ = torch.max(pos_span, dim=2)
        pos_span_min, _ = torch.max(pos_span_min, dim=1)
        pos_span_min = 0 - pos_span_min
        loss_margin_pos = torch.clamp_min(1 - (pos_span_min - qs), 0)
        loss_margin_pos = torch.mean(loss_margin_pos[has_answers_idx])
        loss.append(loss_margin_pos)

        logits = logits.view(-1, seq_len * seq_len)
        labels_batch = labels_batch.view(-1, seq_len * seq_len)
        logits_soft = torch.softmax(logits, dim=1)
        loss_flat = torch.sum(logits_soft * labels_batch, dim=1)
        loss_flat = -torch.log(torch.clamp(loss_flat, 0.0001, 1))
        loss_flat = torch.mean(loss_flat[has_answers_idx], dim=0)
        loss.append(loss_flat)

        return sum(loss) / len(loss)

    def extract_loss(self, logits, span_targets, qs_ext):
        bs, seq_len, seq_len = logits.size()
        qs_ext = qs_ext.view(-1)
        labels_batch = []
        loss = []
        has_answers = []
        global_has_answer = False
        for spans in span_targets:
            label_matrix = [[0] * seq_len for i in range(seq_len)]
            has_answer = 0
            for (beg, end) in spans:
                label_matrix[beg][end] = 1
                has_answer = 1
                global_has_answer = True
            has_answers.append(has_answer)

            labels_batch.append(label_matrix)

        labels_batch = torch.tensor(labels_batch, dtype=torch.float).to(device)
        has_answers = torch.tensor(has_answers, dtype=torch.float).to(device)
        has_answers_idx = has_answers > 0
        neg_span = logits * (1 - labels_batch)
        neg_span_max, _ = torch.max(neg_span, dim=2)
        neg_span_max, _ = torch.max(neg_span_max, dim=1)
        loss_margin_neg = torch.clamp_min(1 - (qs_ext - neg_span_max), 0)
        loss_margin_neg = torch.mean(loss_margin_neg, dim=0)
        loss.append(loss_margin_neg)

        if global_has_answer is False:
            return loss_margin_neg

        pos_span = 0 - logits * labels_batch
        pos_span = pos_span - 10000.0 * (1 - labels_batch)
        pos_span_min, _ = torch.max(pos_span, dim=2)
        pos_span_min, _ = torch.max(pos_span_min, dim=1)
        pos_span_min = 0 - pos_span_min
        loss_margin_pos = torch.clamp_min(1 - (pos_span_min - qs_ext), 0)
        loss_margin_pos = torch.mean(loss_margin_pos[has_answers_idx])
        loss.append(loss_margin_pos)

        logits = logits.view(-1, seq_len * seq_len)
        labels_batch = labels_batch.view(-1, seq_len * seq_len)
        logits_soft = torch.softmax(logits, dim=1)
        loss_flat = torch.sum(logits_soft * labels_batch, dim=1)
        loss_flat = -torch.log(torch.clamp(loss_flat, 0.0001, 1))
        loss_flat = torch.mean(loss_flat[has_answers_idx], dim=0)
        loss += [loss_flat]

        return sum(loss) / len(loss), labels_batch

def get_max_gap(datasets, tokenizer):
    max_gap = 0
    for sample in datasets:
        answers = sample['answers']
        for answer in answers:
            answer_token = tokenizer.tokenize(answer)
            if len(answer_token) > max_gap:
                max_gap = len(answer_token)
    return max_gap + 5

def get_input_feature(features, max_source_length):
    input_texts, span_targets, contexts = [], [], []
    for sample in features:
        context = sample['context']
        if context.strip() == "":
            context = 'context'
        contexts.append(context)
        question = sample['question']
        answers_idx = sample['answers_idx']
        answers_idx = sorted(answers_idx, key=lambda x: x[0])
        span_targets.append(answers_idx)
        input_texts.append((question, context))

    encoding = tokenizer(input_texts,
                         padding='longest',
                         max_length=max_source_length,
                         truncation=True,
                         return_tensors="pt",
                         return_offsets_mapping=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    token_type_ids = encoding.token_type_ids.to(device)
    offset_mapping = encoding['offset_mapping']
    offset_mapping = offset_mapping.tolist()

    offset_mapping_contexts = []
    question_ranges, context_ranges = [], []
    subword_targets = []
    for offset_mapping_item, span_targets_item, context in zip(offset_mapping, span_targets, contexts):
        end1, end2 = -1, -1
        for i, (token_beg, token_end) in enumerate(offset_mapping_item):
            if i == 0:
                continue
            if token_beg == 0 and token_end == 0 and end1 == -1:
                end1 = i
            elif token_beg == 0 and token_end == 0 and end1 > -1:
                end2 = i
                break

        assert end1 >= 1
        if end2 == -1:
            end2 = len(offset_mapping_item) - 1
        question_ranges.append((1, end1 - 1))
        assert end1 + 1 < end2 - 1
        context_ranges.append((end1 + 1, end2 - 1))
        offset_mapping_context_item = offset_mapping_item[end1 + 1: end2]
        offset_mapping_contexts.append(offset_mapping_context_item)
        a_idx = 0
        subword_targets_item = []
        beg_idx_selected, end_idx_selected = -1, -1
        for i, (token_beg, token_end) in enumerate(offset_mapping_context_item):
            if a_idx >= len(span_targets_item):
                break
            beg_idx_search, end_idx_search = span_targets_item[a_idx]
            if token_beg <= beg_idx_search and beg_idx_search < token_end and beg_idx_selected == -1:
                beg_idx_selected = i
            if token_beg <= end_idx_search and end_idx_search <= token_end:
                end_idx_selected = i
                assert beg_idx_selected <= end_idx_selected
                subword_targets_item.append([beg_idx_selected, end_idx_selected])
                a_idx += 1
                beg_idx_selected, end_idx_selected = -1, -1

        subword_targets.append(subword_targets_item)

    return input_ids, token_type_ids, attention_mask, context_ranges, question_ranges, offset_mapping_contexts, subword_targets


def subwordid_to_text(batch_example, spans_predict, token_idx_maps, results, golds_answers):
    for sample, spans_p, token_idx_map in zip(batch_example, spans_predict, token_idx_maps):
        context = sample['context']
        id = sample['id']
        answers_item = []
        for beg, end in spans_p:
            word_idx_beg, _ = token_idx_map[beg]
            _, word_idx_end = token_idx_map[end]
            answer = context[word_idx_beg: word_idx_end]
            assert answer != ""
            answers_item.append(answer)
        results[id] = answers_item
        golds_answers[id] = sample['answers']

@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, max_len):
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    golds_answers, results = {}, {}
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        input_ids, token_type_ids, attention_mask, context_ranges, question_ranges, offset_mapping_contexts,\
        subword_targets = get_input_feature(
            batch_example, max_source_length=max_len)
        spans_predict = model(input_ids, token_type_ids, attention_mask, context_ranges)
        subwordid_to_text(batch_example, spans_predict, offset_mapping_contexts, results, golds_answers)
    results_cp = {}
    keys = results.keys()
    for key in keys:
        results_cp[key] = [item for item in results[key]]
    result_score = None
    if golds_answers is not None:
        result_score = multi_span_evaluate(copy.deepcopy(results), copy.deepcopy(golds_answers))
        result_score = {
            'em_f1': result_score['em_f1'],
            'overlap_f1': result_score['overlap_f1']
        }

    return result_score, results_cp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",
                        default='0',
                        type=str)
    parser.add_argument("--model_name",
                        default='bert-base-uncased',
                        # default='microsoft/deberta-v3-base',
                        type=str)
    parser.add_argument("--dataset_name",
                        default='MultiSpanQA',
                        type=str)
    parser.add_argument("--dataset_split",
                        default='in_house',
                        type=str)
    parser.add_argument("--vanilla",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    vanilla = args.vanilla
    force_answer = True
    if args.dataset_name == 'MultiSpanQA':
        force_answer = False
    only_eval = args.only_eval
    dim2 = 64
    debug = args.debug
    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]
    config_name = f'{args.dataset_name}/{model_name_abb}/'



    parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                     f'_ga_{args.gradient_accumulation_steps}'
    output_model_path = f'./outputs/{config_name}/{parameter_name}/'
    path_save_result = f'./results/{config_name}/{parameter_name}/'


    data_path_base = f'./data/{args.dataset_split}/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/valid.json'
    data_path_test = f'{data_path_base}/test.json'

    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)
    if args.dataset_name == 'QUOREF':
        read_dataset = read_quoref
    else:
        read_dataset = read_msqa

    if 'deberta' in model_name_abb.lower():
        Tokenizer = DebertaTokenizerFast
    elif 'albert' in model_name_abb.lower():
        Tokenizer = AlbertTokenizerFast
    else:
        Tokenizer = BertTokenizerFast

    train_examples = read_dataset(data_path_train)
    dev_examples = read_dataset(data_path_dev)
    test_examples = read_dataset(data_path_test)
    if debug:
        train_examples = train_examples[:20]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_span_gap = get_max_gap(train_examples, tokenizer)
    model = SpanQualifier(args.model_name, max_span_gap, dim2, args.max_len, vanilla).to(device)

    print('# parameters:', sum(param.numel() for param in model.parameters()))
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      "vanilla": vanilla,
                      'gradient_accumulation_steps': args.gradient_accumulation_steps,
                      "epoch": args.epoch_num,
                      "train_path": data_path_train,
                      "dev_path": data_path_dev,
                      "test_path": data_path_test,
                      "train_size": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'max_len': args.max_len,
                      'output_model_path': output_model_path,
                      'path_save_result': path_save_result,
                      'init_checkpoint': args.init_checkpoint,
                      'max_span_gap':max_span_gap}, indent=2))
    if args.init_checkpoint:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k in list(model_dict.keys()):
            name = k
            if k.startswith('module.bert.bert.'):
                name = k.replace("module.bert.", "")
            new_state_dict[name] = model_dict[k]
            del model_dict[k]
        model.load_state_dict(new_state_dict, False)
        print('init from:', init_checkpoint)

    if only_eval:
        result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
        print('result_score_dev:', result_score_dev)
        save_dataset(path_save_result + '/dev.json', results_dev)

        result_score_test, results_test = evaluate(model, test_examples, args.eval_batch_size, args.max_len)
        print('result_score_test:', result_score_test)
        save_dataset(path_save_result + '/test.json', results_test)
        exit(0)

    warm_up_ratio = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=int(warm_up_ratio * (t_total)),
                                                num_training_steps=t_total)
    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0


    if args.init_checkpoint:
        result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
        print('best_dev_result:', result_score_dev)
        best_dev_acc = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
    else:
        best_dev_acc = 0

    best_dev_result, best_test_result = None, None
    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        if early_stop>=5:
            break
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        model.train()
        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            input_ids, token_type_ids, attention_mask, context_ranges, question_ranges, offset_mapping_contexts, subword_targets = get_input_feature(
                batch_example, max_source_length=args.max_len)
            try:
                loss = model(input_ids, token_type_ids, attention_mask, context_ranges, targets=subword_targets)
                loss = loss.mean()
                tr_loss += loss.item()
                nb_tr_steps += 1
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            except Exception as e:
                print('error:',e)

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4)) + f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)
        result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
        f1 = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
        print(result_score_dev)
        if f1 > best_dev_acc:
            early_stop = 0
            best_dev_result = result_score_dev
            best_dev_acc = f1
            save_model(output_model_path, model, optimizer)
            save_dataset(path_save_result + '/dev.json', results_dev)
            print('save new best')
            result_score_test, results_test = evaluate(model, test_examples, args.eval_batch_size, args.max_len)
            best_test_result = result_score_test
            print('test:', result_score_test)
            save_dataset(path_save_result + '/test.json', results_test)

    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)
