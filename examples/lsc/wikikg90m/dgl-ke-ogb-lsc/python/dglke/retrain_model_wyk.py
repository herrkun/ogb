import json
import os
from dglke.my_train import ArgParser
from dglke.dataloader import EvalDataset
from dglke.train_pytorch import load_model, test_mp, valid_mp, train_mp
from dglke.dataloader import get_dataset
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
from .dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--save_path', type=str, default='/disk4/ogb/link_level/',
                          help='save path to load model')
        self.add_argument('--gpu', type=int, default=[0, 1, 2, 3], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--num_proc', type=int, default=4,
                          help='The number of processes to train the model in parallel.' \
                               'In multi-GPU training, the number of processes by default is set to match the number of GPUs.' \
                               'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')

def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))

def set_logger(args):
    '''
    Write logs to console and log file
    '''
    log_file = os.path.join(args.save_path, 'load_model.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
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

def main():
    sys_args = CommonArgParser().parse_args()
    save_path = sys_args.save_path
    #save_path = '/disk4/ogb/link_level/zdy/res2/ComplEx_wikikg90m_concat_d_100_g_3.00/'
    config_f = os.path.join(save_path, 'config.json')
    with open(config_f, "r") as f:
        config = json.loads(f.read())
        # config = json.load(f)
    print(config)
    if not config.get('topk'):
        config['topk'] = 10
    #args = ArgParser().parse_args()
    #args = ArgParser().parse_known_args(config)
    args = Namespace(**config)
    args.num_proc = sys_args.num_proc
    args.gpu = sys_args.gpu
    args.test = True
    print(args)
    set_global_seed(args.seed)
    set_logger(args)

    from .dataloader.retrain_embd_dataset import KGDatasetWikiEnsemble_retrain
    dataset = KGDatasetWikiEnsemble_retrain(args)

    model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                       dataset.relation_feat.shape[1])
    #if args.encoder_model_name in ['roberta', 'concat']:
    model.entity_feat.emb = dataset.entity_feat
    model.relation_feat.emb = dataset.relation_feat
    model.evaluator = WikiKG90MEvaluator()
    device = get_device(args)

    #model.transform_net.load_state_dict(th.load(os.path.join(save_path, args.dataset+"_"+args.model_name+"_mlp"),
    #                                            map_location=device)['transform_state_dict'])
    model.load_emb(args.save_path, args.dataset)
    model.share_memory()

    train_data = TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)

    if args.num_proc > 1:
        train_samplers = []
        for i in range(args.num_proc):
            print("Building training sampler for proc %d" % i)
            t1 = time.time()
            # for each GPU, allocate num_proc // num_GPU processes
            train_sampler_head = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='head',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='tail',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                                  args.neg_sample_size, args.neg_sample_size,
                                                                  True, dataset.n_entities,
                                                                  args.has_edge_importance))
            print("Training sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))

        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities,
                                                        args.has_edge_importance)


    eval_dataset = EvalDataset(dataset, args)

    valid_sampler_tails = []
    for i in range(args.num_proc):
        print("creating valid sampler for proc %d"%i)
        t1 = time.time()
        # valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
        #                                                   args.neg_sample_size_eval,
        #                                                   args.neg_sample_size_eval,
        #                                                   args.eval_filter,
        #                                                   mode='head',
        #                                                   num_workers=args.num_workers,
        #                                                   rank=i, ranks=args.num_proc)
        valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                          args.neg_sample_size_eval,
                                                          args.neg_sample_size_eval,
                                                          args.eval_filter,
                                                          mode='tail',
                                                          num_workers=args.num_workers,
                                                          rank=i, ranks=args.num_proc)
        # valid_sampler_heads.append(valid_sampler_head)
        valid_sampler_tails.append(valid_sampler_tail)
        print("Valid sampler for proc %d created, it takes %s seconds"%(i, time.time()-t1))

    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            # test_sampler_heads = []
            for i in range(args.num_test_proc):
                print("creating test sampler for proc %d" % i)
                t1 = time.time()
                # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                #                                                  args.neg_sample_size_eval,
                #                                                  args.neg_sample_size_eval,
                #                                                  args.eval_filter,
                #                                                  mode='head',
                #                                                  num_workers=args.num_workers,
                #                                                  rank=i, ranks=args.num_test_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='tail',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                # test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)
                print("Test sampler for proc %d created, it takes %s seconds" % (i, time.time() - t1))


    procs = []
    barrier = mp.Barrier(args.num_proc)

    start = time.time()
    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None

    if args.num_proc > 1:
        procs = []
        barrier = mp.Barrier(args.num_proc)
        for i in range(args.num_proc):
            # valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
            # test_sampler = [test_sampler_heads[i], test_sampler_tails[i]] if args.test else None
            valid_sampler = [valid_sampler_tails[i]] if args.valid else None
            test_sampler = [test_sampler_tails[i]] if args.test else None
            proc = mp.Process(target=train_mp, args=(args,
                                                     model,
                                                     train_samplers[i],
                                                     valid_sampler,
                                                     test_sampler,
                                                     i,
                                                     rel_parts,
                                                     cross_rels,
                                                     barrier,
                                                     ))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

if __name__ == '__main__':
    main()