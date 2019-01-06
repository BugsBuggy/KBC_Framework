import numpy as np
import sys
sys.path.append("../") # Adds higher directory to python modules path.
from config import *


class activation_function:
    @staticmethod
    def create_activation_function():
        ent_func = None
        if hasattr(activation_function, Config.ent_func):
            ent_func = getattr(activation_function, Config.ent_func)
        return ent_func

    def identity(input):
        return input


def load_model_embeddings(path,prefix="",postfix=""):
    entity_embedding = path+"/"+prefix+ "E" + postfix + ".dat"
    relation_embedding = path+"/"+prefix+ "R" + postfix + ".dat"
    return np.loadtxt(entity_embedding).astype(np.float32), np.loadtxt(relation_embedding).astype(np.float32)


def read_triplets(file, coder):
    triplets = []

    with open(file, "r") as fin:
            if coder is None:
                for l in fin:
                    s, r, o = l.split()
                    triplets.append((s, r, o))
            else:
                for l in fin:
                    s, r, o = l.split()
                    if (coder.entity2idx.get(s) is not None) and (coder.relation2idx.get(r) is not None) \
                            and (coder.entity2idx.get(o) is not None):
                        triplets.append((s, r, o))
    return triplets


class Coder:
    def __init__(self):
        self.freebase2words = {}
        self.entity2idx = {}
        self.relation2idx = {}
        self.idx2relation = {}
        self.idx2entity = {}


    def construct_encoder(self, triplets):
        encoded_train_data = []
        next_ent_cnt = next_rel_cnt = 0

        for sub, rel, obj in triplets:

            sub_id, next_ent_cnt = self._encode_a_concept(sub, next_ent_cnt, self.entity2idx)
            rel_id, next_rel_cnt = self._encode_a_concept(rel, next_rel_cnt, self.relation2idx)
            obj_id, next_ent_cnt = self._encode_a_concept(obj, next_ent_cnt, self.entity2idx)

            encoded_train_data.append((sub_id, rel_id, obj_id))

        return encoded_train_data

    def _encode_a_concept(self, concept, next_cnt, mapper):

        if mapper.get(concept) is None:
            mapper[concept] = next_cnt
            next_cnt += 1

        return mapper[concept], next_cnt









