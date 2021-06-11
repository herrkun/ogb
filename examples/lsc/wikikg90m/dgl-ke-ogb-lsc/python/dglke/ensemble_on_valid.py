import json
import os
from dglke.my_train import ArgParser
from dglke.dataloader import EvalDataset
from dglke.train_pytorch import load_model, test_mp, ensemble_mp
from dglke.dataloader import get_dataset
from dglke.models.ensemble_valid_models import EnsembleValidModel
from dglke.models.ensemble_valid_models_nfeat import EnsembleValidWithNodeFeatModel
from dglke.train_pytorch import test
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import sys as _sys
from argparse import Namespace
import torch as th
import numpy as np
import time
import torch.multiprocessing as mp
import logging
import argparse
from dglke.dataloader.ensemble_dataset import KGDatasetWikiEnsemble
import dglke.ensemble_on_train as ys

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--save_path', type=str,
                          default='/disk4/ogb/link_level/zdy/res2/ensemble/',
                          help='path to save ensemble model')
        self.add_argument('--m1_path', type=str, default='/disk4/ogb/link_level/zdy/res2/ComplEx_wikikg90m_concat_d_100_g_3.00/',
                          help='path to load model 1')
        self.add_argument('--m2_path', type=str, default='/disk4/ogb/link_level/zdy/res2/TransE_l2_wikikg90m_concat_d_100_g_10.00/',
                          help='path to load model 2')
        self.add_argument('--data_path', type=str,
                          default='data',
                          help='path to load data')
        self.add_argument('--m1_bs', type=int,
                          default=149999,
                          help='model 1 best step')
        self.add_argument('--m2_bs', type=int,
                          default=399999,
                          help='model 2 best step')
        self.add_argument('--gpu', type=int, default=[0,1,2,3], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--dim', type=int, default=64,
                          help='The dim of hidden layer.')
        self.add_argument('-g', '--gamma', type=float, default=3.0,
                          help='The margin value in the score function.')
        self.add_argument('--m1_version', type=str, default='ensemble',
                          help='The version of model 1')
        self.add_argument('--m2_version', type=str, default='ensemble',
                          help='The version of model 2')
        self.add_argument('--mlp_version', type=str, default='v1',
                          help='The version of mlp')
        self.add_argument('--embed_version', type=str, default='concat',
                          help='The version of mlp')
        self.add_argument('--mlp_lr', type=float, default=0.0001,
                          help='The learning rate of optimizing mlp')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='double entitiy dim for complex number')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='double relation dim for complex number')
        self.add_argument('--debug', action='store_true',
                          help='debug mode')
        self.add_argument('--rerun', action='store_true',
                          help='score list exist')
        self.add_argument('--dataset', type=str, default='wikikg90m',
                          help='The name of the builtin knowledge graph. Currently, it only supports wikikg90m')

        self.add_argument('--batch_size_eval', type=int, default=None,
                          help='evaluate batch size')

        self.add_argument('--num_proc', type=int, default=4,
                          help='The number of processes to train the model in parallel.' \
                               'In multi-GPU training, the number of processes by default is set to match the number of GPUs.' \
                               'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('-uvn', '--use_valid_nfeat', action='store_true',
                          help='Double relation dim for complex number or canonical polyadic. It is used by RotatE and SimplE'
        )
        self.add_argument('--train_mode', type=str, default='valid',
                          help='The version of mlp')
        self.add_argument('--force_sync_interval', type=int, default=100,
                          help='We force a synchronization between processes every x steps for' \
                               'multiprocessing training. This potentially stablizes the training process'
                               'to get a better performance. For multiprocessing training, it is set to 100 by default.')
        self.add_argument('--epoch', type=int, default=10,
                          help='epoch for ensemble')
        self.add_argument('--k', type=int, default=None,
                          help='return topk candidate')

def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))

def set_logger(save_path):
    '''
    Write logs to console and log file
    '''
    log_file = os.path.join(save_path, 'ensemble.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    #if args.print_on_screen:
    if True:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def set_global_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic=True


def load_result(path, num_proc, step):
    rank_list = []
    ans_list = []
    for i in range(num_proc):
        res = th.load(os.path.join(path, f"valid_{i}_{step}.pkl"))
        ans_list = th.cat([ans_list, res['h,r->t']['t_correct_index']]) if i>0 else res['h,r->t']['t_correct_index']
        rank_list = th.cat([rank_list, res['h,r->t']['t_pred_top10'].cpu()]) if i>0 else res['h,r->t']['t_pred_top10'].cpu()
        score_list = th.cat([rank_list, res['h,r->t']['t_pred_scores'].cpu()]) if i>0 else res['h,r->t']['t_pred_scores'].cpu()
    print(ans_list.shape)
    print(rank_list.shape)
    return ans_list, rank_list, score_list

def eval_rerank(ans_list, rank_list):
    res = {
        'h,r->t': {
            't_correct_index': ans_list,
            't_pred_top10': rank_list
        }
    }
    from ogb.lsc import WikiKG90MEvaluator
    evaluator = WikiKG90MEvaluator()
    result_dict = evaluator.eval(res)
    print(res)
    print(result_dict['mrr'])
    return res, result_dict

def get_args(sys_args, path):
    config_f = os.path.join(path, 'config.json')
    with open(config_f, "r") as f:
        config = json.loads(f.read())
        # config = json.load(f)
    print(config)
    if sys_args.k is not None:
        config['topk'] = sys_args.k
    else:
        if not config.get('topk'):
            config['topk'] = 10
    return Namespace(**config)

def load_exist_model(args, dataset):
    model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                       dataset.relation_feat.shape[1])
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat
    model.evaluator = WikiKG90MEvaluator()
    model.load_emb(args.save_path, args.dataset)
    if args.num_proc > 1 or args.async_update:
        model.share_memory()
    return model

def load_exist_ensemble_model(args, dataset):
    model = ys.load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                       dataset.other_entity_feat.shape[1], dataset.relation_feat.shape[1])
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat
        model.other_entity_feat.emb = dataset.other_entity_feat
    model.evaluator = WikiKG90MEvaluator()
    model.load_emb(args.save_path, args.dataset)
    if args.num_proc > 1 or args.async_update:
        model.share_memory()
    return model

def main():
    sys_args = CommonArgParser().parse_args()
    args_m1 = get_args(sys_args, sys_args.m1_path)
    if sys_args.batch_size_eval is not None:
        args_m1.batch_size_eval = sys_args.batch_size_eval
    print(args_m1)
    set_global_seed(args_m1.seed)

    args_m2 = get_args(sys_args, sys_args.m1_path)
    print(args_m2)
    print(sys_args.dataset)

    '''dataset = get_dataset(args_m1.data_path,
                          args_m1.dataset,
                          args_m1.format,
                          args_m1.delimiter,
                          args_m1.data_files,
                          args_m1.has_edge_importance)
    '''
    dataset = KGDatasetWikiEnsemble(sys_args)
    if not sys_args.use_valid_nfeat:
        ensemble_model = EnsembleValidModel(sys_args, sys_args.embed_version, "ensemble_valid_model", dataset.n_entities,
                                            dataset.n_relations, sys_args.dim, sys_args.gamma,
                                            double_entity_emb=sys_args.double_ent, double_relation_emb=sys_args.double_rel,
                                            ent_feat_dim=dataset.entity_feat.shape[1],
                                            other_ent_feat_dim=dataset.other_entity_feat.shape[1],
                                            rel_feat_dim=dataset.relation_feat.shape[1])
        ensemble_model.entity_feat.emb = dataset.entity_feat
        ensemble_model.other_entity_feat.emb = dataset.other_entity_feat
        ensemble_model.relation_feat.emb = dataset.relation_feat
        ensemble_model.share_memory()
    else:
        ensemble_model = EnsembleValidWithNodeFeatModel(
            sys_args, sys_args.embed_version, "ensemble_valid_model", dataset.n_entities,
            dataset.n_relations, sys_args.dim, sys_args.gamma,
            double_entity_emb=sys_args.double_ent, double_relation_emb=sys_args.double_rel,
            ent_feat_dim=dataset.entity_feat.shape[1],
            other_ent_feat_dim=dataset.other_entity_feat.shape[1],
            rel_feat_dim=dataset.relation_feat.shape[1],
            other_valid_nfeat_dim=dataset.other_nfeat_valid.shape[1]
        )
        ensemble_model.entity_feat.emb = dataset.entity_feat
        ensemble_model.other_entity_feat.emb = dataset.other_entity_feat
        ensemble_model.relation_feat.emb = dataset.relation_feat
        ensemble_model.other_valid_nfeat.emb = dataset.other_nfeat_valid
        ensemble_model.share_memory()

    if sys_args.rerun:
        model_1 = None
        model_2 = None
    else:
        if sys_args.m1_version == 'ensemble':
            model_1 = load_exist_ensemble_model(args_m1, dataset)
        else:
            model_1 = load_exist_model(args_m1, dataset)
        if sys_args.m2_version == 'ensemble':
            model_2 = load_exist_ensemble_model(args_m2, dataset)
        else:
            model_2 = load_exist_model(args_m2, dataset)

    '''
    ans_list_m1, rank_list_m1, score_list_m1 = load_result(args_m1.save_path,args_m1.num_proc,sys_args.m1_bs)
    print(len(ans_list_m1), len(rank_list_m1))
    ans_list_m2, rank_list_m2, score_list_m2 = load_result(args_m2.save_path, args_m2.num_proc, sys_args.m2_bs)
    print(len(ans_list_m2), len(rank_list_m2))
    print((ans_list_m1 != ans_list_m2).nonzero())
    eval_rerank(ans_list_m1, rank_list_m1)
    eval_rerank(ans_list_m2, rank_list_m2)
    '''
    eval_dataset = EvalDataset(dataset, args_m1)
    set_logger(sys_args.save_path)
    valid_sampler_tails = []
    num_proc = sys_args.num_proc
    for i in range(num_proc):
        print("creating valid sampler for proc %d" % i)
        t1 = time.time()
        valid_sampler_tail = eval_dataset.create_sampler('valid', args_m1.batch_size_eval,
                                                         args_m1.neg_sample_size_eval,
                                                         args_m1.neg_sample_size_eval,
                                                         args_m1.eval_filter,
                                                         mode='tail',
                                                         num_workers=args_m1.num_workers,
                                                         rank=i, ranks=num_proc)
        valid_sampler_tails.append(valid_sampler_tail)
        print("Valid sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))
    procs = []
    barrier = mp.Barrier(num_proc)
    # mp.set_start_method('spawn')
    for i in range(num_proc):
        valid_sampler = [valid_sampler_tails[i]]
        proc = mp.Process(target=ensemble_mp, args=(args_m1,
                                                 args_m2,
                                                 model_1,
                                                 model_2,
                                                 sys_args,
                                                 valid_sampler,
                                                 i,
                                                 barrier,
                                                 ensemble_model
                                                 ))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    print("finish!")
    #ensemble_sampler = create_sampler(dataset, )

if __name__ == '__main__':
    main()
