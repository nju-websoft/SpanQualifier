# coding=utf-8
from transformers import get_linear_schedule_with_warmup, T5Tokenizer, BartTokenizer, T5Config
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from tqdm import trange
import os
import random
from utils import save_dataset, set_seed, save_model, read_dataset
import json
import argparse
import time
from torch import nn
import copy
from tqdm import tqdm
from eval_script import multi_span_evaluate, get_entities
import ast
import numpy as np
import torch

device = torch.device("cuda:0")
class SpanQualifier(nn.Module):
    def __init__(self, model_path):
        super(SpanQualifier, self).__init__()
        self.t5_model = ConditionalGeneration.from_pretrained(model_path)
        # dim = self.t5_model.config.d_model
        n_gpu = torch.cuda.device_count()
        layer_num = self.t5_model.config.num_layers
        layer_per_gpu = layer_num // n_gpu
        layer_per_gpu_remainder = layer_num % n_gpu
        device_map = {}
        cur_layer = 0
        for n in range(n_gpu):
            device_map[n] = []
            if n < layer_per_gpu_remainder:
                layer_assigned = layer_per_gpu + 1
            else:
                layer_assigned = layer_per_gpu

            for i in range(layer_assigned):
                device_map[n].append(cur_layer)
                cur_layer += 1
        self.t5_model.parallelize(device_map)

    def forward(self, input_ids, input_masks, labels=None):
        if labels is not None:
            t5_output = self.t5_model(input_ids=input_ids,
                                      attention_mask=input_masks,
                                      labels=labels,
                                      return_dict=True)
            loss = t5_output.loss
            return loss
        else:
            enc_time_beg = time.time()

            enc_time_end = time.time()
            dec_time_beg = time.time()
            t5_output = self.t5_model.generate(
                input_ids=input_ids,
                # encoder_outputs=ModelOutput(last_hidden_state=encoder_q),
                max_length=100,
                attention_mask=input_masks,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False
            )
            output_sequences = t5_output.sequences
            # score_list = t5_output.score_list
            predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            predicts = predicts[0].split(split_symbol)
            dec_time_end = time.time()
            return predicts, enc_time_end - enc_time_beg, dec_time_end - dec_time_beg

def get_input_feature(features, tokenizer, max_length):
    input_list, target_list = [], []
    for b_i, sample in enumerate(features):
        question = sample['question']
        if use_context:
            context = sample['context']
            input_list.append(f'Question: {question} Context: {context}')
        else:
            input_list.append(f'Question: {question}')
        answers = copy.deepcopy(sample['answers'])
        assert len(answers) > 0
        answer = split_symbol.join(answers)
        target_list.append(answer)

    input_ids, input_masks = tokenizer_fun(input_list, max_length)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)

    labels, _ = tokenizer_fun(target_list, max_length)
    labels = [
        [label if label != tokenizer.pad_token_id else -100 for label in labels_example] for labels_example in
        labels
    ]
    labels = torch.tensor(np.asarray(labels), dtype=torch.long).to(device)
    return input_ids, input_masks, labels

def tokenizer_fun(input_ids, max_len):
    encoding = tokenizer(input_ids,
                         padding='longest',
                         max_length=max_len,
                         truncation=True)
    ids = encoding.input_ids
    mask = encoding.attention_mask
    return ids, mask

@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, tokenizer, max_len):
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    preds = {}
    golds = {}
    dataset_gold = []
    time_all_enc, time_all_dec = 0, 0
    time_all = 0
    assert eval_batch_size == 1
    for sample in tqdm(test_examples):

        input_ids, input_masks, _ = get_input_feature([sample], tokenizer, max_len)
        beg = time.time()
        spans_predicts, enc_time, dec_time = model(input_ids, input_masks)
        # print(spans_predicts)
        if use_context:
            context = sample['context']
            spans_predicts_new = []
            for spans_predict in spans_predicts:
                if spans_predict.lower().strip() in context.lower():
                    spans_predicts_new.append(spans_predict)
            if len(spans_predicts_new) != 0:
                spans_predicts = spans_predicts_new
                spans_predicts = list(set(spans_predicts))

        end = time.time()
        time_all += (end-beg)
        time_all_enc += enc_time
        time_all_dec += dec_time
        id = sample['id']
        answers = sample['answers']
        preds[id] = spans_predicts
        sample['pred'] = spans_predicts
        golds[id] = answers
        dataset_gold.append({
            'id': id,
            'question': sample['question'],
            'answers': answers,
            'pred': spans_predicts
        })
    print('enc avg:', round(time_all_enc * 100 / len(test_examples), 2))
    print('dec avg:', round(time_all_dec * 100 / len(test_examples), 2))
    print('time_all:', round(time_all * 100 / len(test_examples), 2))
    print('Throughout:', round(len(test_examples) / time_all, 2))
    scores = evaluate_fun(copy.deepcopy(preds), copy.deepcopy(golds), brief=True)
    return scores, preds, dataset_gold


def read_msqa(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        if 'label' not in sample:
            dataset = dataset_init
            break
        id = sample['id']
        question = sample['question']
        context = sample['context']
        label = sample['label']

        answers = get_entities(label, context)
        answers = [answer[0] for answer in answers]
        assert len(answers) >= 2
        dataset.append(
            {
                'id': id,
                'context': ' '.join(context),
                'question': ' '.join(question),
                'answers': answers
            }
        )

    return dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='t5-base',
                        type=str)
    parser.add_argument("--sample_negative",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval_train",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--gpu",
                        default="1",
                        type=str)
    parser.add_argument("--dataset_name",
                        default='MultiSpanQA',
                        type=str)
    parser.add_argument("--dataset_split",
                        default='in_house',
                        # default='official',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--ga',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--results_save_path",
                        default='results',
                        type=str)
    parser.add_argument("--output_dir",
                        default='outputs',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=ast.literal_eval)
    parser.add_argument("--use_context",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--save_model",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--acc_epoch",
                        default=-1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    split_symbol = ' # '

    only_eval = args.only_eval
    only_eval_train = args.only_eval_train
    debug = args.debug
    save_model_flag = args.save_model
    model_name = args.model_name
    use_context = args.use_context
    if 'bart' in model_name:
        Tokenizer = BartTokenizer
        ConditionalGeneration = BartForConditionalGeneration
    else:
        ConditionalGeneration = T5ForConditionalGeneration
        Tokenizer = T5Tokenizer
    evaluate_fun = multi_span_evaluate
    dataset_name = args.dataset_name
    read_dataset_fun = read_dataset
    read_dataset_fun = read_msqa
    data_path_base = f'./data/in_house/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'

    data_path_valid = f'{data_path_base}/valid.json'
    data_path_test = f'{data_path_base}/test.json'

    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]

    if use_context:
        config_name = f'{args.dataset_name}/Sequence_context/{model_name_abb}'
    else:
        config_name = f'{args.dataset_name}/Sequence/{model_name_abb}/'

    parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                     f'_ga_{args.ga}'
    output_model_path = f'./{args.output_dir}/{config_name}/{parameter_name}/'
    path_save_result = f'./{args.results_save_path}/{config_name}/{parameter_name}/'
    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)

    if debug:
        train_examples = read_dataset_fun(data_path_train)[:10]
        dev_examples = read_dataset_fun(data_path_valid)[:10]
        test_examples = read_dataset_fun(data_path_test)[:10]
    else:
        train_examples = read_dataset_fun(data_path_train)
        dev_examples = read_dataset_fun(data_path_valid)
        test_examples = read_dataset_fun(data_path_test)

    train_batch_size = args.train_batch_size // args.ga
    tokenizer = Tokenizer.from_pretrained(args.model_name)


    model = SpanQualifier(args.model_name)#.to(device)
    vocab_size = model.t5_model.config.vocab_size
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      'ga': args.ga,
                      'init': args.init,
                      "epoch": args.epoch_num,
                      'save_model':save_model_flag,
                      "train_path": data_path_train,
                      "dev_path": data_path_valid,
                      "test_path": data_path_test,
                      "train_size": len(train_examples),
                      "train_examples": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'max_len': args.max_len,
                      'output_model_path': output_model_path,
                      'use_context': use_context,
                      'path_save_result': path_save_result,
                      'init_checkpoint': args.init_checkpoint}, indent=2))
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    if only_eval or only_eval_train:
        args.init = True

    if args.init and args.init_checkpoint is None:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', init_checkpoint)
    elif args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)

    if only_eval_train:
        scores, results_train, readable_results_train = evaluate(model, train_examples, args.eval_batch_size, tokenizer, args.max_len)
        print(f'train:', scores)
        save_dataset(data_path_base, 'train_pred.json', train_examples)
        exit(0)

    if only_eval:

        scores, results_valid, readable_results_valid = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                             args.max_len)
        print('dev:', scores)
        save_dataset(path_save_result, '/valid.json', results_valid)
        save_dataset(path_save_result, '/readable_valid.json', readable_results_valid)

        scores, results_test, readable_results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                                               args.max_len)
        print('test:', scores)
        save_dataset(path_save_result, '/test.json', results_test)
        save_dataset(path_save_result, '/readable_test.json', readable_results_test)
        exit(0)

    warm_up_ratio = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                # num_warmup_steps=int(warm_up_ratio * (t_total)),
                                                num_warmup_steps=1000,
                                                num_training_steps=t_total)
    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    best_test_acc = 0
    best_dev_acc = 0
    best_dev_result, best_test_result = None, None
    if args.init_checkpoint is not None:
        scores_valid, results_valid, readable_results_valid = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                                 args.max_len)
        scores = sum([scores_valid[key] for key in scores_valid.keys()])
        print('scores_dev:', scores_valid)
        best_dev_acc = scores

    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
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
            input_ids, input_masks, labels = get_input_feature(batch_example, tokenizer, args.max_len)
            # beg = time.time()
            loss = model(input_ids, input_masks, labels)
            # end = time.time()
            # print(end - beg)
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss = loss / args.ga
            loss.backward()
            if (step + 1) % args.ga == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4)) + f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)

        # if epoch >= 16:
        if epoch >= args.acc_epoch:
            scores_valid, results_valid, readable_results_valid = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                                     args.max_len)
            print('dev:', scores_valid)
            scores = sum([scores_valid[key] for key in scores_valid.keys()])
            if scores > best_dev_acc:
                best_dev_acc = scores
                print('save new best')
                if save_model_flag:
                    save_model(output_model_path, model, optimizer)
                else:
                    save_dataset(path_save_result, '/valid.json', results_valid)
                    save_dataset(path_save_result, '/readable_valid.json', readable_results_valid)

                    scores_test, results_test, readable_results_test = evaluate(model, test_examples, args.eval_batch_size,
                                                                                tokenizer,
                                                                                args.max_len)
                    print('test:', scores_test)
                    save_dataset(path_save_result, '/test.json', results_test)
                    save_dataset(path_save_result, '/readable_test.json', readable_results_test)
    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

    ###############################
    if save_model_flag:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', init_checkpoint)
        scores, results_valid, readable_results_valid = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                             args.max_len)
        print('dev:', scores)
        save_dataset(path_save_result, '/valid.json', results_valid)
        save_dataset(path_save_result, '/readable_valid.json', readable_results_valid)

        scores, results_test, readable_resultas_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                                               args.max_len)
        print('test:', scores)
        save_dataset(path_save_result, '/test.json', results_test)
        save_dataset(path_save_result, '/readable_test.json', readable_results_test)

