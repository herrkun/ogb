# -*- coding: utf-8 -*-
#
# general_models.py
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
"""
Graph Embedding Model
1. TransE
2. TransR
3. RESCAL
4. DistMult
5. ComplEx
6. RotatE
7. SimplE
"""
import os
import numpy as np
import math
import dgl.backend as F
import pdb
import torch
import torch.nn as nn

backend = os.environ.get('DGLBACKEND', 'pytorch')
from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import abs
from .pytorch.tensor_models import masked_select
from .pytorch.tensor_models import get_device, get_dev
from .pytorch.tensor_models import norm
from .pytorch.tensor_models import get_scalar
from .pytorch.tensor_models import reshape
from .pytorch.tensor_models import cuda
from .pytorch.tensor_models import ExternalEmbedding
from .pytorch.tensor_models import InferEmbedding
from .pytorch.score_fun import *
from .pytorch.loss import LossGenerator

DEFAULT_INFER_BATCHSIZE = 2048

EMB_INIT_EPS = 2.0

class MLP(torch.nn.Module):
    def __init__(self, input_entity_dim, entity_dim, input_relation_dim, relation_dim, version='v1'):
        super(MLP, self).__init__()
        self._version = version
        if self._version == 'v1':
            self.transform_e_net = torch.nn.Linear(input_entity_dim, entity_dim)
            self.transform_r_net = torch.nn.Linear(input_relation_dim, relation_dim)
            self.transform_v_net = torch.nn.Sequential(
                torch.nn.Linear(entity_dim*2+relation_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2),
                torch.nn.Tanh(),
            )

        elif self._version == 'v2':
            self.transform_e_net = torch.nn.Sequential(
                torch.nn.Linear(input_entity_dim, entity_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(entity_dim, entity_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(entity_dim, entity_dim)
            )
            self.transform_r_net = torch.nn.Sequential(
                torch.nn.Linear(input_relation_dim, relation_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(relation_dim, relation_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(entity_dim, entity_dim)
            )
        else:
            self.transform_e_net = torch.nn.Sequential(
                torch.nn.Linear(input_entity_dim, entity_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(entity_dim, entity_dim),
                torch.nn.Tanh()
            )
            self.transform_r_net = torch.nn.Sequential(
                torch.nn.Linear(input_relation_dim, relation_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(relation_dim, relation_dim),
                torch.nn.Tanh()
            )
        self.reset_parameters()

    def embed_entity(self, embeddings):
        return self.transform_e_net(embeddings)

    def embed_valid(self, embeddings):
        return self.transform_v_net(embeddings)

    def embed_relation(self, embeddings):
        return self.transform_r_net(embeddings)

    def reset_parameters(self):
        if self._version == 'v1':
            nn.init.xavier_uniform_(self.transform_r_net.weight)
            nn.init.xavier_uniform_(self.transform_e_net.weight)
            nn.init.xavier_uniform_(self.transform_v_net[0].weight)
            nn.init.xavier_uniform_(self.transform_v_net[2].weight)

        elif self._version == 'v2':
            nn.init.xavier_uniform_(self.transform_r_net[0].weight)
            nn.init.xavier_uniform_(self.transform_r_net[2].weight)
            nn.init.xavier_uniform_(self.transform_e_net[0].weight)
            nn.init.xavier_uniform_(self.transform_e_net[2].weight)
        else:
            nn.init.xavier_uniform_(self.transform_r_net[0].weight)
            nn.init.xavier_uniform_(self.transform_r_net[2].weight)
            nn.init.xavier_uniform_(self.transform_r_net[4].weight)
            nn.init.xavier_uniform_(self.transform_e_net[0].weight)
            nn.init.xavier_uniform_(self.transform_e_net[2].weight)
            nn.init.xavier_uniform_(self.transform_e_net[4].weight)


class EnsembleValidModel(object):

    def __init__(self, args, encoder_model_name, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False, ent_feat_dim=-1, other_ent_feat_dim=-1
                 , rel_feat_dim=-1):
        super(EnsembleValidModel, self).__init__()
        self.args = args
        self.has_edge_importance = False
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / hidden_dim
        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim
        self.encoder_model_name = encoder_model_name
        device = get_device(args)
        if self.encoder_model_name == 'concat':
            self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim, F.cpu())
            self.entity_feat = ExternalEmbedding(args, n_entities, ent_feat_dim,
                                                 F.cpu(), is_feat=True)
            self.other_entity_feat = ExternalEmbedding(args, n_entities, other_ent_feat_dim,
                                                       F.cpu(), is_feat=True)
            rel_dim = relation_dim
            self.use_mlp = True
            self.transform_net = MLP(entity_dim + ent_feat_dim + other_ent_feat_dim,
                                     entity_dim, relation_dim + rel_feat_dim, relation_dim, args.mlp_version)
            self.rel_dim = rel_dim
            self.entity_dim = entity_dim
            self.strict_rel_part = False
            self.soft_rel_part = False
            print(self.strict_rel_part, self.soft_rel_part)
            assert not self.strict_rel_part and not self.soft_rel_part
            self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim, F.cpu())
            self.relation_feat = ExternalEmbedding(args, n_relations, rel_feat_dim, F.cpu(), is_feat=True)
        else:
            self.entity_feat = ExternalEmbedding(args, n_entities, ent_feat_dim,
                                                 F.cpu(), is_feat=True)
            self.other_entity_feat = ExternalEmbedding(args, n_entities, other_ent_feat_dim,
                                                       F.cpu(), is_feat=True)
            rel_dim = relation_dim
            self.use_mlp = True
            self.transform_net = MLP(ent_feat_dim + other_ent_feat_dim,
                                     entity_dim, rel_feat_dim, relation_dim, args.mlp_version)
            self.rel_dim = rel_dim
            self.entity_dim = entity_dim
            self.strict_rel_part = False
            self.soft_rel_part = False
            print(self.strict_rel_part, self.soft_rel_part)
            assert not self.strict_rel_part and not self.soft_rel_part
            self.relation_feat = ExternalEmbedding(args, n_relations, rel_feat_dim, F.cpu(), is_feat=True)

        self.reset_parameters()

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process embeddings access.
        """

        self.entity_feat.share_memory()
        self.other_entity_feat.share_memory()
        self.relation_feat.share_memory()
        self.transform_net.share_memory()
        if self.encoder_model_name == 'concat':
            self.entity_emb.share_memory()
            self.relation_emb.share_memory()

    def save_emb(self, path, dataset):
        """Save the model.

        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        torch.save({'transform_state_dict': self.transform_net.state_dict()},
                       os.path.join(path, dataset + "_" + self.model_name + "_mlp"))
        if self.encoder_model_name == 'concat':
            self.relation_emb.save(path, dataset + '_' + self.model_name + '_relation')
            self.entity_emb.save(path, dataset + '_' + self.model_name + '_entity')

    def load_emb(self, path, dataset):
        """Load the model.

        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset + '_' + self.model_name + '_entity')
        self.relation_emb.load(path, dataset + '_' + self.model_name + '_relation')

    def reset_parameters(self):
        """Re-initialize the model.
        """
        if self.encoder_model_name == 'concat':
            self.entity_emb.init(self.emb_init)
            self.relation_emb.init(self.emb_init)
        self.transform_net.reset_parameters()

    def get_score(self, query, candidate, score1, score2, gpu_id):
        if self.encoder_model_name == 'concat':
            tails = self.transform_net.embed_entity(th.cat(
                [self.entity_feat(candidate.view(-1), gpu_id, False),
                 self.other_entity_feat(candidate.view(-1), gpu_id, False),
                 self.entity_emb(candidate.view(-1), gpu_id, False)], -1))
            heads = self.transform_net.embed_entity(th.cat(
                [self.entity_feat(query[:, 0], gpu_id, False),
                 self.other_entity_feat(query[:, 0], gpu_id, False),
                 self.entity_emb(query[:, 0], gpu_id, False)], -1))
            rels = self.transform_net.embed_relation(th.cat(
                [self.relation_feat(query[:, 1], gpu_id, False),
                 self.relation_emb(query[:, 1], gpu_id, False)], -1))
        else:
            tails = self.transform_net.embed_entity(th.cat(
                [self.entity_feat(candidate.view(-1), gpu_id, False),
                 self.other_entity_feat(candidate.view(-1), gpu_id, False)], -1))
            heads = self.transform_net.embed_entity(th.cat(
                [self.entity_feat(query[:, 0], gpu_id, False),
                 self.other_entity_feat(query[:, 0], gpu_id, False)], -1))
            rels = self.transform_net.embed_relation(th.cat(
                [self.relation_feat(query[:, 1], gpu_id, False)], -1))
        num_chunks = len(query)  # 512
        chunk_size = 1
        neg_sample_size = candidate.shape[1]  # 1001
        ent_hidden_dim = heads.shape[1]  # 100
        rel_hidden_dim = rels.shape[1]
        heads = heads.reshape(num_chunks, chunk_size, ent_hidden_dim)
        rels = rels.reshape(num_chunks, chunk_size, rel_hidden_dim)
        tails = tails.reshape(num_chunks, neg_sample_size, ent_hidden_dim)
        heads = th.tile(heads, [1, neg_sample_size, 1])
        rels = th.tile(rels, [1, neg_sample_size, 1])
        emb = th.cat([heads, rels, tails], dim=-1)
        weight = self.transform_net.embed_valid(emb)
        score1 = score1.reshape(num_chunks, neg_sample_size, 1)
        score2 = score2.reshape(num_chunks, neg_sample_size, 1)
        #print(weight.shape)
        #print(score1.shape)
        #print(score2.shape)
        score = th.sigmoid(th.sum(weight * th.cat([score1, score2], dim=-1), dim=-1)) # 512 * 1001
        return score
