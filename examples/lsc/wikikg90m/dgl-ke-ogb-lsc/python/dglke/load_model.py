import json
import os
from dglke.my_train import ArgParser
from dglke.dataloader import EvalDataset
from dglke.train_pytorch import load_model
from dglke.dataloader import get_dataset
from dglke.train_pytorch import test
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import sys as _sys
from argparse import Namespace
import torch as th
import numpy as np
def set_global_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic=True

save_path = '/disk4/ogb/link_level/zdy/res/ComplEx_wikikg90m_concat_d_100_g_3.015/'
config_f = os.path.join(save_path, 'config.json')
with open(config_f, "r") as f:
    config = json.loads(f.read())
    # config = json.load(f)
print(config)

#args = ArgParser().parse_args()
#args = ArgParser().parse_known_args(config)
if not config.get('topk'):
    config['topk'] = 10
args = Namespace(**config)
print(args)
set_global_seed(args.seed)
#
dataset = get_dataset(args.data_path,
                      args.dataset,
                      args.format,
                      args.delimiter,
                      args.data_files,
                      args.has_edge_importance)

model = load_model(args, dataset.n_entities, dataset.n_relations, dataset.entity_feat.shape[1],
                   dataset.relation_feat.shape[1])
model.transform_net.load_state_dict(th.load(os.path.join(save_path, args.dataset+"_"+args.model_name+"_mlp"))['transform_state_dict'])
#model.transform_net = model.transform_net.to(th.device('cuda:' + str(0)))

model.entity_feat.emb = dataset.entity_feat
model.relation_feat.emb = dataset.relation_feat
model.evaluator = WikiKG90MEvaluator()
model.load_emb(args.save_path, args.dataset)

eval_dataset = EvalDataset(dataset, args)
valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_eval,
                                                             1,
                                                             args.eval_filter,
                                                             mode='tail',
                                                             num_workers=args.num_workers,
                                                             rank=0, ranks=1)

valid_samplers = [valid_sampler_tail]
step = rank = 0
args.gpu = []
valid_input_dict = test(args, model, valid_samplers, step, rank, mode='Valid')
th.save(valid_input_dict, os.path.join(args.save_path, "my_valid_{}_{}.pkl".format(rank, step)))
