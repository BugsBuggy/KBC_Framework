from config import Config
import numpy as np
from collections import defaultdict as ddict


class Dataset:
    def __init__(self, num_entities, num_relations, splits):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.splits = splits


def index(symbol_freqs, file):
    with open(file, 'w') as f:
        for i, k in symbol_freqs.items():
            f.write(str(k) + '\t' + str(i) + '\n')


def preprocess():
    print('Preprocessing ' + Config.dataset)
    dataset_dir = Config.data_dir + Config.dataset + '/'
    raw = {}

    # read data and collect entity and relation names
    entities = {}
    relations = {}
    ent_id = 0
    rel_id = 0
    for k, file in Config.raw_split_files.items():
        with open(dataset_dir + file, 'r') as f:
            raw[k] = list(map(lambda s: s.strip().split('\t'), f.readlines()))
            for t in raw[k]:

                if t[0] not in entities:
                    entities[t[0]] = ent_id
                    ent_id += 1
                if t[1] not in relations:
                    relations[t[1]] = rel_id
                    rel_id += 1
                if t[2] not in entities:
                    entities[t[2]] = ent_id
                    ent_id += 1
            print(str(len(raw[k])) + ' triples in ' + file)

    # sort entities and relations by frequency
    print(str(len(relations)) + ' distinct relations')
    print(str(len(entities)) + ' distinct entities')
    print('Writing indexes...')
    index(relations, dataset_dir + Config.relation_index_file)
    index(entities, dataset_dir + Config.entity_index_file)

    # write out
    print('Writing triples...')
    for k, file in Config.split_files.items():
       with open(dataset_dir + file, 'w') as f:
           for t in raw[k]:
               f.write(str(entities[t[0]]))
               f.write(',')
               f.write(str(relations[t[1]]))
               f.write(',')
               f.write(str(entities[t[2]]))
               f.write('\n')

    print('Done')


def load():
    print('Loading ' + Config.dataset)
    dataset_dir = Config.data_dir + Config.dataset + '/'
    splits = {}
    max_ent = -1
    max_rel = -1
    for k,v in Config.split_files.items():
        splits[k] = np.loadtxt(dataset_dir + v, delimiter=',', dtype=np.int64)
        max_ent = max(max_ent, np.max(splits[k][:,0]), np.max(splits[k][:,2]))
        max_rel = max(max_rel, np.max(splits[k][:,1]))
    num_entities = max_ent+1
    num_relations = max_rel+1
    return Dataset(num_entities, num_relations, splits)