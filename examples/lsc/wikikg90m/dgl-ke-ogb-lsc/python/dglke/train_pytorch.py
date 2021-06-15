# -*- coding: utf-8 -*-
#
# train_pytorch.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from .models.pytorch.tensor_models import thread_wrapped_func
from .models import KEModel
from .utils import save_model, get_compatible_batch_size
from .models.pytorch.tensor_models import get_device

import os
import logging
import time
from functools import wraps

import dgl
from dgl.contrib import KVClient
import dgl.backend as F

from .dataloader import EvalDataset
from .dataloader import get_dataset
import pdb
from collections import defaultdict
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
from tqdm import tqdm
import pickle
from math import ceil
cuda = lambda arr, gpu: arr.cuda(gpu)

class GammaLossFunction(th.nn.Module):
    def __init__(self, gamma=3.0):
        super(GammaLossFunction,self).__init__()
        self.gamma = gamma
        return

    def forward(self, neg, pos):
        loss = th.mean(th.relu(self.gamma + neg - pos))
        return loss

def load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim, ckpt=None):
    model = KEModel(args, args.encoder_model_name, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel,
                    ent_feat_dim=ent_feat_dim, rel_feat_dim=rel_feat_dim)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model


def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path, ent_feat_dim, rel_feat_dim):
    model = load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim)
    model.load_emb(ckpt_path, args.dataset)
    return model


def save_best_model(args,  model, step, rank, best_mrr):
    evaluator = WikiKG90MEvaluator()
    ans = 0.0
    num_proc = args.num_proc
    for i in range(num_proc):
        res = th.load(os.path.join(args.save_path, "valid_{}_{}.pkl".format(i,step)))
        res['h,r->t']['t_pred_top10'] = res['h,r->t']['t_pred_top10'][:, :10]
        result_dict = evaluator.eval(res)
        ans += result_dict['mrr'] / num_proc
    logging.info("[step {}] now mrr: {:.3f}".format(step, ans))
    if (ans > best_mrr):
        best_mrr = ans
        save_model(args, model, None, None)
        logging.info("[proc {} step {}] model saved".format(rank, step))
    return best_mrr


def train(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.retrain:
        if rank == 0:
            model.transform_net.load_state_dict(th.load(os.path.join(args.save_path,
                                                                     args.dataset + "_" + args.model_name + "_mlp"))[
                                                    'transform_state_dict'])
        if barrier is not None:
            barrier.wait()

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)
    
    if args.encoder_model_name in ['roberta', 'concat']:
        if gpu_id < 0:
            model.transform_net = model.transform_net.to(th.device('cpu'))
        else:
            model.transform_net = model.transform_net.to(th.device('cuda:' + str(gpu_id)))
        #model.transform_net = model.transform_net.to(get_device(args))
        optimizer = th.optim.Adam(model.transform_net.parameters(), args.mlp_lr)
    else:
        optimizer = None

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    best_mrr = 0
    for step in range(0, args.max_step):
        if step % 1000 == 0:
            print(f"rank {rank} start step {step}")
        start1 = time.time()
        pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        if optimizer is not None:
            optimizer.zero_grad()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)
        if optimizer is not None:
            optimizer.step()

        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
            (step + 1) % args.force_sync_interval == 0:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    logging.info('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                logging.info('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                time.time() - start))
                logging.info('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

        # if True:
        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            logging.info('[proc {}]barrier wait in validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            valid_start = time.time()
            if valid_samplers is not None:
                valid_input_dict = test(args, model, valid_samplers, step, rank, mode='Valid')
                th.save(valid_input_dict, os.path.join(args.save_path, "valid_{}_{}.pkl".format(rank, step)))
            if test_samplers is not None:
                test_input_dict = test(args, model, test_samplers, step, rank, mode='Test')
                th.save(test_input_dict, os.path.join(args.save_path, "test_{}_{}.pkl".format(rank, step)))
            logging.info('[proc {}]validation and test take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()
            if rank == 0 and not args.no_save_emb:
                best_mrr = save_best_model(args, model, step, rank, best_mrr)
                logging.info("[step {}] best mrr: {:.3f}".format(step, best_mrr))
            if barrier is not None:
                barrier.wait()

    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    time.sleep(10)
    #if rank == 0 and not args.no_save_emb:
    #    save_model(args, model, None, None)
    #    print('proc {} model saved'.format(rank))

    if barrier is not None:
        barrier.wait()
    print('proc {} after barrier'.format(rank))
    if args.async_update:
        model.finish_async_update()
    print('proc {} finish async update'.format(rank))
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)
    print('proc {} return'.format(rank))

def test(args, model, test_samplers, step, rank=0, mode='Test'):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    # print (test_samplers)
    # pdb.set_trace()
    with th.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        scores = defaultdict(list)
        for sampler in test_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=not args.print_on_screen, total=ceil(sampler.num_edges/sampler.batch_size)):
                log, score = model.forward_test_wikikg(query, ans, candidate, sampler.mode, gpu_id, args.topk)
                logs[sampler.mode].append(log)
                scores[sampler.mode].append(score)
                answers[sampler.mode].append(ans)
        print("[{}] finished {} forward".format(rank, mode))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            input_dict['h,r->t'] = {
                't_correct_index': th.cat(answers['h,r->t'], 0),
                't_pred_top10': th.cat(logs['h,r->t'], 0),
                't_pred_scores': th.cat(scores['h,r->t'], 0)
            }
        # if 't,r->h' in answers:
        #     assert 't,r->h' in logs, "t,r->h not in logs"
        #     input_dict['t,r->h'] = {'h_correct_index': th.cat(answers['t,r->h'], 0), 'h_pred_top10': th.cat(logs['t,r->h'], 0)}
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()
    # test_samplers[0] = test_samplers[0].reset()
    # test_samplers[1] = test_samplers[1].reset()

    return input_dict

@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, test_samplers, rank, rel_parts, cross_rels, barrier)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, 0, rank, mode)

def valid(args, model, valid_samplers=None, rank=0, barrier=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1
    if rank == 0:
        model.transform_net.load_state_dict(th.load(os.path.join(args.save_path,
                                                                 args.dataset + "_" + args.model_name + "_mlp"))['transform_state_dict'])
    #wait load model
    print('gpu_id: ', gpu_id)
    if barrier is not None:
        barrier.wait()
    if args.encoder_model_name in ['roberta', 'concat']:
        if gpu_id < 0:
            model.transform_net = model.transform_net.to(th.device('cpu'))
        else:
            model.transform_net = model.transform_net.to(th.device('cuda:' + str(gpu_id)))
    step = 0
    valid_input_dict = test(args, model, valid_samplers, step, rank, mode='Valid')
    th.save(valid_input_dict, os.path.join(args.save_path, "gpu_valid_{}_{}.pkl".format(rank, step)))

def ensemble(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers=None, rank=0, barrier=None, ensemble_model=None):
    if len(sys_args.gpu) > 0:
        gpu_id = sys_args.gpu[rank % len(sys_args.gpu)]
    else:
        gpu_id = -1
    mode = 'h,r->t'
    k = 10
    index = 0
    debug_mode = sys_args.debug
    if sys_args.rerun:
        score1_list = th.load(os.path.join(sys_args.save_path, "m1_valid_score_{}.pkl".format(rank)),
                              map_location='cuda:' + str(gpu_id))
        score2_list = th.load(os.path.join(sys_args.save_path, "m2_valid_score_{}.pkl".format(rank)),
                              map_location='cuda:' + str(gpu_id))
    else:

        if rank == 0:
            model_1.transform_net.load_state_dict(th.load(os.path.join(args_m1.save_path,
                                                                     args_m1.dataset + "_" + args_m1.model_name + "_mlp"))['transform_state_dict'])
            model_2.transform_net.load_state_dict(th.load(os.path.join(args_m2.save_path,
                                                                       args_m2.dataset + "_" + args_m2.model_name + "_mlp"))[
                                                      'transform_state_dict'])
        #wait load model
        if barrier is not None:
            barrier.wait()
        if gpu_id < 0:
            model_1.transform_net = model_1.transform_net.to(th.device('cpu'))
            model_2.transform_net = model_2.transform_net.to(th.device('cpu'))
        else:
            model_1.transform_net = model_1.transform_net.to(th.device('cuda:' + str(gpu_id)))
            model_2.transform_net = model_2.transform_net.to(th.device('cuda:' + str(gpu_id)))


        with th.no_grad():
            score1_list = defaultdict(list)
            score2_list = defaultdict(list)
            #label_list = defaultdict(list)

            for sampler in valid_samplers:
                print(sampler.num_edges, sampler.batch_size)
                for query, ans, candidate in tqdm(sampler, disable=False,
                                                  total=ceil(sampler.num_edges / sampler.batch_size)):
                    score1 = model_1.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id,
                                                          trace=False)
                    score2 = model_2.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id,
                                                          trace=False)
                    score1_list[mode].append(score1)
                    score2_list[mode].append(score2)
                    index += 1
                    if debug_mode and index > 10:
                        break
                    #label_list[mode].append(label)
        if not debug_mode:
            th.save(score1_list, os.path.join(sys_args.save_path, "m1_valid_score_{}.pkl".format(rank)))
            th.save(score2_list, os.path.join(sys_args.save_path, "m2_valid_score_{}.pkl".format(rank)))

    if ensemble_model is not None:
        if gpu_id < 0:
            ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cpu'))
        else:
            ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cuda:' + str(gpu_id)))
        optimizer = th.optim.Adam(ensemble_model.transform_net.parameters(), sys_args.mlp_lr)
        for tmp_i in range(sys_args.epoch):
            index = 0
            for i in range(len(valid_samplers)):
                valid_samplers[i] = valid_samplers[i].reset()

            for sampler in valid_samplers:
                print(sampler.num_edges, sampler.batch_size)
                for query, ans, candidate in tqdm(sampler, disable=False,
                                                  total=ceil(sampler.num_edges / sampler.batch_size)):
                    score1 = score1_list[mode][index]  # 512 * 1001
                    score2 = score2_list[mode][index]
                    score = ensemble_model.get_score(query, candidate, score1, score2, gpu_id)
                    tmp = th.tile(ans.reshape(-1, 1), [1, candidate.shape[1]]).to(th.device('cuda:' + str(gpu_id)))
                    pos = score.gather(-1, tmp)
                    loss_f = GammaLossFunction(gamma=sys_args.gamma)
                    loss = loss_f(score, pos)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    index += 1
                    if sys_args.force_sync_interval > 0 and \
                            (index + 1) % sys_args.force_sync_interval == 0:
                        print(loss)
                        barrier.wait()
                    if debug_mode and index > 10:
                       break
    if barrier is not None:
        barrier.wait()
    if (rank == 0):
        save_model(sys_args, ensemble_model, None, None)
    if barrier is not None:
        barrier.wait()

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

    logs = defaultdict(list)
    answers = defaultdict(list)
    scores = defaultdict(list)
    index = 0
    with th.no_grad():
        for sampler in valid_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=False, total=ceil(sampler.num_edges/sampler.batch_size)):
                score1 = score1_list[mode][index]
                score2 = score2_list[mode][index]
                if ensemble_model is not None:
                    score = ensemble_model.get_score(query, candidate, score1, score2, gpu_id)
                else:
                    score = 0.5 * score1 + 0.5 * score2
                log = F.argsort(score, dim=1, descending=True)[:, :k]
                score = F.topk(score, k=k, dim=1, descending=True)
                logs[mode].append(log.cpu())
                scores[mode].append(score.cpu())
                answers[mode].append(ans.cpu())
                index += 1
                if debug_mode and index > 10:
                    break

    print("[{}] finished {} forward".format(rank, mode))
    input_dict = {}
    #assert len(answers) == 1
    assert 'h,r->t' in answers
    if 'h,r->t' in answers:
        assert 'h,r->t' in logs, "h,r->t not in logs"
        input_dict['h,r->t'] = {
            't_correct_index': th.cat(answers['h,r->t'], 0),
            't_pred_top10': th.cat(logs['h,r->t'], 0),
            't_pred_scores': th.cat(scores['h,r->t'], 0)
        }
    th.save(input_dict, os.path.join(sys_args.save_path, "ensemble_{}.pkl".format(rank)))

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

    if barrier is not None:
        barrier.wait()



def load_ensemble(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers=None, rank=0, barrier=None, ensemble_model=None):
    if len(sys_args.gpu) > 0:
        gpu_id = sys_args.gpu[rank % len(sys_args.gpu)]
    else:
        gpu_id = -1
    mode = 'h,r->t'
    if sys_args.k is not None:
        k = sys_args.k
    else:
        k = 10
    index = 0
    debug_mode = sys_args.debug
    if rank == 0:
        ensemble_model.transform_net.load_state_dict(th.load(os.path.join(sys_args.save_path,
                                                                          sys_args.dataset + "_" + "ensemble_valid_model" + "_mlp"))[
                                                         'transform_state_dict'])
    if barrier is not None:
        barrier.wait()
    if gpu_id < 0:
        ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cpu'))
    else:
        ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cuda:' + str(gpu_id)))

    if sys_args.rerun:
        score1_list = th.load(os.path.join(sys_args.save_path, "m1_{}_score_{}.pkl".format(sys_args.mode,rank)),
                              map_location='cuda:' + str(gpu_id))
        score2_list = th.load(os.path.join(sys_args.save_path, "m2_{}_score_{}.pkl".format(sys_args.mode,rank)),
                              map_location='cuda:' + str(gpu_id))
    else:
        if rank == 0:
            model_1.transform_net.load_state_dict(th.load(os.path.join(args_m1.save_path,
                                                                     args_m1.dataset + "_" + args_m1.model_name + "_mlp"))['transform_state_dict'])
            model_2.transform_net.load_state_dict(th.load(os.path.join(args_m2.save_path,
                                                                       args_m2.dataset + "_" + args_m2.model_name + "_mlp"))[
                                                      'transform_state_dict'])

        #wait load model
        if barrier is not None:
            barrier.wait()
        if gpu_id < 0:
            model_1.transform_net = model_1.transform_net.to(th.device('cpu'))
            model_2.transform_net = model_2.transform_net.to(th.device('cpu'))
        else:
            model_1.transform_net = model_1.transform_net.to(th.device('cuda:' + str(gpu_id)))
            model_2.transform_net = model_2.transform_net.to(th.device('cuda:' + str(gpu_id)))

        with th.no_grad():
            score1_list = defaultdict(list)
            score2_list = defaultdict(list)
            #label_list = defaultdict(list)

            for sampler in valid_samplers:
                print(sampler.num_edges, sampler.batch_size)
                for query, ans, candidate in tqdm(sampler, disable=False,
                                                  total=ceil(sampler.num_edges / sampler.batch_size)):
                    score1 = model_1.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id,
                                                          trace=False)
                    score2 = model_2.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id,
                                                          trace=False)
                    score1_list[mode].append(score1)
                    score2_list[mode].append(score2)
                    index += 1
                    if debug_mode and index > 10:
                        break
                    #label_list[mode].append(label)
        if not debug_mode:
            th.save(score1_list, os.path.join(sys_args.save_path, "m1_{}_score_{}.pkl".format(sys_args.mode,rank)))
            th.save(score2_list, os.path.join(sys_args.save_path, "m2_{}_score_{}.pkl".format(sys_args.mode,rank)))

    if barrier is not None:
        barrier.wait()

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

    logs = defaultdict(list)
    answers = defaultdict(list)
    scores = defaultdict(list)
    index = 0
    with th.no_grad():
        for sampler in valid_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=False, total=ceil(sampler.num_edges/sampler.batch_size)):
                score1 = score1_list[mode][index]
                score2 = score2_list[mode][index]
                if ensemble_model is not None:
                    score = ensemble_model.get_score(query, candidate, score1, score2, gpu_id)
                else:
                    score = 0.5 * score1 + 0.5 * score2
                log = F.argsort(score, dim=1, descending=True)[:, :k]
                score = F.topk(score, k=k, dim=1, descending=True)
                logs[mode].append(log.cpu())
                scores[mode].append(score.cpu())
                answers[mode].append(ans.cpu())
                index += 1
                if debug_mode and index > 10:
                    break

    print("[{}] finished {} forward".format(rank, mode))
    input_dict = {}
    #assert len(answers) == 1
    assert 'h,r->t' in answers
    if 'h,r->t' in answers:
        assert 'h,r->t' in logs, "h,r->t not in logs"
        input_dict['h,r->t'] = {
            't_correct_index': th.cat(answers['h,r->t'], 0),
            't_pred_top10': th.cat(logs['h,r->t'], 0),
            't_pred_scores': th.cat(scores['h,r->t'], 0)
        }
    th.save(input_dict, os.path.join(sys_args.save_path, "load_ensemble_{}_{}.pkl".format(sys_args.mode,rank)))

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

def ensemble_test(sys_args, valid_samplers=None, rank=0, barrier=None, ensemble_model=None):
    if len(sys_args.gpu) > 0:
        gpu_id = sys_args.gpu[rank % len(sys_args.gpu)]
    else:
        gpu_id = -1

        #to do
    mode = 'h,r->t'
    k = 10
    index = 0
    debug_mode = sys_args.debug

    if gpu_id < 0:
        ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cpu'))
    else:
        ensemble_model.transform_net = ensemble_model.transform_net.to(th.device('cuda:' + str(gpu_id)))
    print(sys_args.mlp_lr)
    optimizer = th.optim.Adam(ensemble_model.transform_net.parameters(), sys_args.mlp_lr)

    for tmp_i in range(sys_args.epoch):
        for i in range(len(valid_samplers)):
            valid_samplers[i] = valid_samplers[i].reset()


        for sampler in valid_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=False,
                                              total=ceil(sampler.num_edges / sampler.batch_size)):
                score = ensemble_model.get_score(query, candidate, gpu_id) #512*1001

                tmp = th.tile(ans.reshape(-1,1),[1,candidate.shape[1]]).to(th.device('cuda:' + str(gpu_id)))
                pos = score.gather(-1, tmp)
                loss_f = GammaLossFunction(gamma=sys_args.gamma)
                loss = loss_f(score, pos)
                
                '''
                label = th.nn.functional.one_hot(ans, num_classes=candidate.shape[1]).to(th.device('cuda:' + str(gpu_id)))
                label = label.reshape(-1)  # 512512
                score = score.reshape(-1)
                score = th.stack([1 - score, score], dim=-1).float()
                loss_f = th.nn.CrossEntropyLoss()
                loss = loss_f(score, label)
                '''

                loss.backward()
                #for parms in  ensemble_model.transform_net.parameters():
                #    print(parms)
                optimizer.step()
                optimizer.zero_grad()
                index += 1
                if sys_args.force_sync_interval > 0 and \
                        (index + 1) % sys_args.force_sync_interval == 0:
                    print(loss)
                    barrier.wait()
                if debug_mode and index > 10:
                   break

    if barrier is not None:
        barrier.wait()

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

    logs = defaultdict(list)
    answers = defaultdict(list)
    scores = defaultdict(list)
    index = 0
    with th.no_grad():
        for sampler in valid_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=False, total=ceil(sampler.num_edges/sampler.batch_size)):
                score = ensemble_model.get_score(query, candidate, gpu_id)
                #print(score.shape)
                log = F.argsort(score, dim=1, descending=True)[:, :k]
                #print(log.shape)
                score = F.topk(score, k=k, dim=1, descending=True)
                #print(score.shape)
                #print(ans.shape)
                logs[mode].append(log)
                scores[mode].append(score)
                answers[mode].append(ans)
                index += 1
                if debug_mode and index > 10:
                    break

    print("[{}] finished {} forward".format(rank, mode))
    input_dict = {}
    #assert len(answers) == 1
    assert 'h,r->t' in answers
    if 'h,r->t' in answers:
        assert 'h,r->t' in logs, "h,r->t not in logs"
        input_dict['h,r->t'] = {
            't_correct_index': th.cat(answers['h,r->t'], 0),
            't_pred_top10': th.cat(logs['h,r->t'], 0),
            't_pred_scores': th.cat(scores['h,r->t'], 0)
        }
    th.save(input_dict, os.path.join(sys_args.save_path, "ensemble_{}.pkl".format(rank)))

    for i in range(len(valid_samplers)):
        valid_samplers[i] = valid_samplers[i].reset()

    if barrier is not None:
        barrier.wait()
    if (rank == 0):
        save_model(sys_args, ensemble_model, None, None)


@thread_wrapped_func
def valid_mp(args, model, valid_samplers, rank=0, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    valid(args, model, valid_samplers, rank, barrier)

@thread_wrapped_func
def ensemble_mp(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers, rank=0, barrier=None, ensemble_model=None):
    if sys_args.num_proc > 1:
        th.set_num_threads(1)
    ensemble(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers, rank, barrier, ensemble_model)

@thread_wrapped_func
def load_ensemble_mp(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers, rank=0, barrier=None, ensemble_model=None):
    if sys_args.num_proc > 1:
        th.set_num_threads(1)
    load_ensemble(args_m1, args_m2, model_1, model_2, sys_args, valid_samplers, rank, barrier, ensemble_model)


@thread_wrapped_func
def ensemble_test_mp(sys_args, valid_samplers, rank=0, barrier=None, ensemble_model=None):
    if sys_args.num_proc > 1:
        th.set_num_threads(1)
    ensemble_test(sys_args, valid_samplers, rank, barrier, ensemble_model)
