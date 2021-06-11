from ogb.lsc import WikiKG90MDataset
from .KGDataset import KGDataset
import numpy as np
import os.path as osp



class WikiKG90MDatasetEnsemble(WikiKG90MDataset):

    def __init__(self, root: str = 'dataset'):
        super(WikiKG90MDatasetEnsemble, self).__init__(root)
        self._other_entity_feat = None
        self._other_nfeat_valid = None
        self._other_nfeat_test = None
        self._train_val_hrt = None
        self._train_fewer_hrt = None
        self._train_upsample_hrt = None
        self._train_hrt = np.concatenate((self._train_hrt, np.load('/disk4/ogb/link_level/dataset/wikikg90m_kddcup2021/processed/trian_val_topk_add_h.npy')))

    @property
    def train_val_hrt(self) -> np.ndarray:
        '''

        '''
        if self._train_val_hrt is None:
            path2 = osp.join(self.processed_dir, 'val_hrt_wyk.npy')
            path3 = '/disk4/ogb/link_level/dataset/wikikg90m_kddcup2021/processed/upsample_on_val_wyk.npy'
            self._train_val_hrt = np.concatenate((self._train_hrt, np.load(path2), np.load(path3)))
            print("Training dataset with validation have %d samples" % self._train_val_hrt.shape[0])
        return self._train_val_hrt

    @property
    def train_fewer_hrt(self) -> np.ndarray:
        '''
            using fewer train data for training
        '''
        if self._train_fewer_hrt is None:
            # path = '/disk4/ogb/link_level/dataset/metapath_feat/top1_tail_filter_pairre_train.npy'
            path = '/disk4/ogb/link_level/dataset/wikikg90m_kddcup2021/processed/generator_new_trian.npy'
            self._train_fewer_hrt = np.load(path)
            print("Training dataset with filter have %d samples" % self._train_fewer_hrt.shape[0])
        return self._train_fewer_hrt

    @property
    def train_upsample_hrt(self) -> np.ndarray:
        '''
            using upsample train data for training
        '''
        if self._train_upsample_hrt is None:
            path1 = '/disk4/ogb/link_level/dataset/metapath_feat/top1_tail_filter_pairre_train.npy'
            path2 = '/disk4/ogb/link_level/dataset/wikikg90m_kddcup2021/processed/generator_new_trian.npy'
            self._train_upsample_hrt = np.concatenate((self._train_hrt, np.load(path1), np.load(path2)))
            print("Training dataset with filter have %d samples" % self._train_upsample_hrt.shape[0])
        return self._train_upsample_hrt

    @property
    def num_feat_dims(self) -> int:
        '''
            Dimensionality of relation and entity features obtained by roberta
        '''
        return 200

    @property
    def entity_feat(self) -> np.ndarray:
        '''
            Entity feature
            - np.ndarray of shape (num_entities, num_feat_dims)
              i-th row stores the feature of i-th entity
              * Loading everything into memory at once
              * saved in np.float16
        '''
        if self._entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            self._entity_feat = np.load(path, mmap_mode='r')
        return self._entity_feat

    @property
    def other_entity_feat(self) -> np.ndarray:
        if self._other_entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            # path = '/disk4/ogb/link_level/dataset/metapath_feat/node_metapath_feat.npy'
            self._other_entity_feat = np.load(path, mmap_mode='r')
        return self._other_entity_feat

    @property
    def other_nfeat_valid(self) -> np.ndarray:
        if self._other_nfeat_valid is None:
            path = osp.join(self.processed_dir, 'val_cand_occur_feat2.npy')
            self._other_nfeat_valid = np.load(path, mmap_mode='r')
        return self._other_nfeat_valid

    @property
    def other_nfeat_test(self) -> np.ndarray:
        if self._other_nfeat_test is None:
            path = osp.join(self.processed_dir, 'test_cand_occur_feat.npy')
            self._other_nfeat_test = np.load(path, mmap_mode='r')
        return self._other_nfeat_test

    @property
    def other_nfeat_train(self) -> np.ndarray:
        if self._other_nfeat_test is None:
            path = osp.join(self.processed_dir, 'train_cand_occur_feat.npy')
            self._other_nfeat_test = np.load(path, mmap_mode='r')
        return self._other_nfeat_test

    @property
    def all_entity_feat(self) -> np.ndarray:
        if self._all_entity_feat is None:
            path = osp.join(self.original_root, 'entity_feat.npy')
            # path = "/disk4/ogb/link_level/dataset/metapath_feat/node_metapath_feat.npy"
            self._all_entity_feat = np.load(path)
        return self._all_entity_feat


class WikiKG90MDatasetEnsembleTrainNFeat(WikiKG90MDataset):

    def __init__(self, root: str = 'dataset'):
        super(WikiKG90MDatasetEnsembleTrainNFeat, self).__init__(root)
        self._other_entity_feat = None
        self._other_nfeat_valid = None
        self._other_nfeat_test = None

    @property
    def num_feat_dims(self) -> int:
        '''
            Dimensionality of relation and entity features obtained by roberta
        '''
        return 200

    @property
    def entity_feat(self) -> np.ndarray:
        '''
            Entity feature
            - np.ndarray of shape (num_entities, num_feat_dims)
              i-th row stores the feature of i-th entity
              * Loading everything into memory at once
              * saved in np.float16
        '''
        if self._entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            self._entity_feat = np.load(path, mmap_mode='r')
        return self._entity_feat

    @property
    def other_entity_feat(self) -> np.ndarray:
        if self._other_entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            # path = '/disk4/ogb/link_level/dataset/metapath_feat/node_metapath_feat.npy'
            self._other_entity_feat = np.load(path, mmap_mode='r')
        return self._other_entity_feat

    @property
    def other_nfeat_valid(self) -> np.ndarray:
        if self._other_nfeat_valid is None:
            path = osp.join(self.processed_dir, 'valid_nfeat.npy')
            self._other_nfeat_valid = np.load(path, mmap_mode='r')
        return self._other_nfeat_valid

    @property
    def other_nfeat_test(self) -> np.ndarray:
        if self._other_nfeat_test is None:
            path = osp.join(self.processed_dir, 'test_nfeat.npy')
            self._other_nfeat_test = np.load(path, mmap_mode='r')
        return self._other_nfeat_test

    @property
    def other_nfeat_train(self) -> np.ndarray:
        if self._other_nfeat_test is None:
            path = osp.join(self.processed_dir, 'train_nfeat.npy')
            self._other_nfeat_test = np.load(path, mmap_mode='r')
        return self._other_nfeat_test

    @property
    def all_entity_feat(self) -> np.ndarray:
        if self._all_entity_feat is None:
            path = osp.join(self.original_root, 'entity_feat.npy')
            # path = "/disk4/ogb/link_level/dataset/metapath_feat/node_metapath_feat.npy"
            self._all_entity_feat = np.load(path)
        return self._all_entity_feat

class KGDatasetWikiEnsembleNFeat(KGDataset):
    '''Load a knowledge graph FB15k

    The FB15k dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, sys_args, name='wikikg90m'):
        self.name = name
        path = "/disk4/ogb/link_level/dataset/"
        self.dataset = WikiKG90MDatasetEnsembleTrainNFeat(path)
        self.train = self.dataset.train_hrt.T
        self.n_entities = self.dataset.num_entities
        self.n_relations = self.dataset.num_relations
        self.valid = None
        self.test = None
        self.valid_dict = self.dataset.valid_dict
        self.test_dict = self.dataset.test_dict
        self.entity_feat = self.dataset.entity_feat
        self.relation_feat = self.dataset.relation_feat
        # self.other_entity_feat_train = self.dataset.other_entity_feat_train
        self.other_nfeat_train = self.dataset.other_nfeat_train
        self.other_nfeat_valid = self.dataset.other_nfeat_valid
        self.other_nfeat_test = self.dataset.other_nfeat_test
        print(f'sys_args.use_valid_nfeat: {sys_args.use_valid_nfeat}, sys_args.train_mode: {sys_args.train_mode}')

        self.other_nfeat_train = self.dataset.other_nfeat_train
        self.other_nfeat_valid = self.dataset.other_nfeat_valid
        self.other_nfeat_test = self.dataset.other_nfeat_test
        if 't,r->h' in self.valid_dict:
            del self.valid_dict['t,r->h']
        if 't,r->h' in self.test_dict:
            del self.valid_dict['t,r->h']

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None



class KGDatasetWikiEnsemble(KGDataset):
    '''Load a knowledge graph FB15k

    The FB15k dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, sys_args, name='wikikg90m'):
        self.name = name
        path = "/disk4/ogb/link_level/dataset/"
        self.dataset = WikiKG90MDatasetEnsemble(path)
        if sys_args.train_with_val:
            self.train = self.dataset.train_val_hrt.T
        elif sys_args.train_fewer:
            self.train = self.dataset.train_fewer_hrt.T
        elif sys_args.train_upsample:
            self.train = self.dataset.train_upsample_hrt.T
        else:
            self.train = self.dataset.train_hrt.T
        self.n_entities = self.dataset.num_entities
        self.n_relations = self.dataset.num_relations
        self.valid = None
        self.test = None
        self.valid_dict = self.dataset.valid_dict
        self.test_dict = self.dataset.test_dict
        self.entity_feat = self.dataset.entity_feat
        self.relation_feat = self.dataset.relation_feat
        self.other_entity_feat = self.dataset.other_entity_feat
        print(f'sys_args.use_valid_nfeat: {sys_args.use_valid_nfeat}, sys_args.train_mode: {sys_args.train_mode}')
        if sys_args.use_valid_nfeat:
            if sys_args.train_mode == 'valid':
                print('use features on validation')
                self.other_nfeat_valid = self.dataset.other_nfeat_valid
            else:
                print('use features on test')
                self.other_nfeat_valid = self.dataset.other_nfeat_test
        else:
            self.other_nfeat_valid = None
        if 't,r->h' in self.valid_dict:
            del self.valid_dict['t,r->h']
        if 't,r->h' in self.test_dict:
            del self.valid_dict['t,r->h']

    @property
    def emap_fname(self):
        return None

    @property
    def rmap_fname(self):
        return None
