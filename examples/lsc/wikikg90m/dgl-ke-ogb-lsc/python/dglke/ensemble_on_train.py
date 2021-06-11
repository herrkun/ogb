import os
import logging
import time

from .dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
from .dataloader import get_dataset

from .utils import get_compatible_batch_size, save_model, CommonArgParser

backend = os.environ.get('DGLBACKEND', 'pytorch')
assert backend.lower() == 'pytorch'
import torch
import numpy as np
import torch.multiprocessing as mp
from .train_pytorch import train, train_mp
from .train_pytorch import test, test_mp
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
from .train import set_global_seed, set_logger
from .models.ensemble_ke_models import EnsembleKEModel


def load_model(args, n_entities, n_relations, ent_feat_dim, other_ent_feat_dim, rel_feat_dim, ckpt=None):
    model = EnsembleKEModel(args, "concat", args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel,
                    ent_feat_dim=ent_feat_dim, other_ent_feat_dim=other_ent_feat_dim, rel_feat_dim=rel_feat_dim)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--train_with_val', action='store_true',
                          help='whether include valditation when training.')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.'\
                                  'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')

        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument('--encoder_model_name', type=str, default='shallow',
                          help='shallow or roberta or concat')
        self.add_argument('--mlp_lr', type=float, default=0.0001,
                          help='The learning rate of optimizing mlp')
        self.add_argument('--seed', type=int, default=0,
                          help='random seed')
        self.add_argument('--topk', type=int, default=10,
                          help='The number of topk in valid and test')
        self.add_argument('--mlp_version', type=str, default='v1',
                          help='The version of mlp')
        self.add_argument('-uvn', '--use_valid_nfeat', action='store_true',
                          help='Double relation dim for complex number or canonical polyadic. It is used by RotatE and SimplE'
        )
        self.add_argument('--train_mode', type=str, default='valid',
                          help='The version of mlp')

def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_{}_d_{}_g_{}'.format(args.model_name, args.dataset, args.encoder_model_name, args.hidden_dim, args.gamma)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
    assert args.dataset == 'wikikg90m'
    args.neg_sample_size_eval = 1000
    set_global_seed(args.seed)

    init_time_start = time.time()
    # load dataset and samplers
    # dataset = get_dataset(args.data_path,
    #                       args.dataset,
    #                       args.format,
    #                       args.delimiter,
    #                       args.data_files,
    #                       args.has_edge_importance)
    from .dataloader.ensemble_dataset import KGDatasetWikiEnsemble
    dataset = KGDatasetWikiEnsemble(args)
    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
            'The number of processes needs to be divisible by the number of GPUs'
    # For multiprocessing training, we need to ensure that training processes are synchronized periodically.
    if args.num_proc > 1:
        args.force_sync_interval = 1000

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    args.soft_rel_part = args.mix_cpu_gpu and args.rel_part
    print ("To build training dataset")
    t1 = time.time()
    train_data = TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
    print ("Training dataset built, it takes %d seconds " %(time.time( ) -t1))
    # if there is no cross partition relaiton, we fall back to strict_rel_part
    args.strict_rel_part = args.mix_cpu_gpu and (train_data.cross_part == False)
    args.num_workers = 8 # fix num_worker to 8
    args.num_thread = 1 # fix num_thread to 1
    set_logger(args)
    with open(os.path.join(args.save_path, args.encoder_model_name), 'w') as f:
        f.write(args.encoder_model_name)

    eval_dataset = EvalDataset(dataset, args)
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
    else: # This is used for debug
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities,
                                                        args.has_edge_importance)
    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        print("To create eval_dataset")
        t1 = time.time()

        print("eval_dataset created, it takes %d seconds" % (time.time() - t1))

    if args.valid:
        if args.num_proc > 1:
            # valid_sampler_heads = []
            valid_sampler_tails = []
            for i in range(args.num_proc):
                print("creating valid sampler for proc %d " %i)
                t1 = time.time()
                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='tail',
                                                                 num_workers=args.num_workers,
                                                                 rank=i, ranks=args.num_proc)
                valid_sampler_tails.append(valid_sampler_tail)
                print("Valid sampler for proc %d created, it takes %s seconds " %(i, time.time( ) -t1))
        else:
            valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_eval,
                                                             1,
                                                             args.eval_filter,
                                                             mode='tail',
                                                             num_workers=args.num_workers,
                                                             rank=0, ranks=1)
    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            # test_sampler_heads = []
            for i in range(args.num_test_proc):
                print("creating test sampler for proc %d " %i)
                t1 = time.time()
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='tail',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                test_sampler_tails.append(test_sampler_tail)
                print("Test sampler for proc %d created, it takes %s seconds " %(i, time.time( ) -t1))
        else:
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            1,
                                                            args.eval_filter,
                                                            mode='tail',
                                                            num_workers=args.num_workers,
                                                            rank=0, ranks=1)
    print("To create model")
    t1 = time.time()
    # model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
    model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                       dataset.other_entity_feat.shape[1], dataset.relation_feat.shape[1])
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.other_entity_feat.emb = dataset.other_entity_feat
        model.relation_feat.emb = dataset.relation_feat
    print("Model created, it takes %s seconds" % (time.time( ) -t1))
    model.evaluator = WikiKG90MEvaluator()

    if args.num_proc > 1 or args.async_update:
        model.share_memory()

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    # train
    start = time.time()
    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None

    if args.num_proc > 1:
        procs = []
        barrier = mp.Barrier(args.num_proc)
        for i in range(args.num_proc):
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
    else:
        valid_samplers = [valid_sampler_tail] if args.valid else None
        test_samplers = [test_sampler_tail] if args.test else None
        train(args, model, train_sampler, valid_samplers, test_samplers, rel_parts=rel_parts)

    print('training takes {} seconds'.format(time.time() - start))

if __name__ == '__main__':
    main()