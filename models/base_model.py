import torch
import numpy as np
import os
import torch.nn as nn
from config import Config
from util.helpers import *
# move model imports to end of file or else there might occur circular dependencies

class BaseModel(nn.Module):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super().__init__()
        # input: list of indices, output: corresponding word embeddings
        self.entity_emb = nn.Embedding(num_entities, entity_emb_size, sparse=True)
        self.relation_emb = nn.Embedding(num_relations, entity_emb_size, sparse=True)
        self.ent_func = ent_func
        self.entity_emb_size = entity_emb_size
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_weights = self.entity_emb.weight
        self.relation_weights = self.relation_emb.weight

    # factory method for creating models
    @staticmethod
    def createModel(num_entities, num_relations):
        try:
            ent_func = activation_function.create_activation_function()
            if ent_func is None:
                ent_func = eval("torch.nn.%s" % Config.ent_func)
        except AttributeError as a:
            print("Activation function is not defined")
        try:
            model = globals()[Config.model](num_entities, num_relations, Config.dimensions, ent_func)
        except NameError as n:
            print("Model is not defined!")
        return model

    # puts lookup table and embedding weights to specified device
    def weights_to_device(self, device):
        if  device == "cpu":
            self = self.cpu()
        elif self.entity_emb.weight.device is not device:
            self.entity_emb = self.entity_emb.to(device)
            self.relation_emb = self.relation_emb.to(device)
            self.entity_weights = self.entity_emb.weight.to(device)
            self.relation_weights = self.relation_emb.weight.to(device)


    # important: put weights to cpu or it will not work with gpu
    def dump_embeddings(self, dir, postfix=""):
        ent_file = 'E' + postfix + '.dat'
        rel_file = 'R' + postfix + '.dat'
        self.entity_emb.weight.data = self.ent_func(self.entity_emb.weight.data)
        if not os.path.exists(os.getcwd() + "/" + dir):
            os.makedirs(os.getcwd() + "/" + dir)
        print("dumping embeddings...")
        np.savetxt(os.getcwd() + "/" + dir + "/" + ent_file, self.entity_emb.weight.cpu().data.numpy())
        np.savetxt(os.getcwd() + "/" + dir + "/" + rel_file, self.relation_emb.weight.cpu().data.numpy())


    ## x is ndarray holding batch triples (positive or negative)
    ## this produces raw scores -> can be fed in ranking loss or softmax etc
    def forward(self, x):
        e1 = self.entity_emb(x[:, 0])
        r = self.relation_emb(x[:, 1])
        e2 = self.entity_emb(x[:, 2])
        return self.forward_emb(e1, r, e2)

    # As done in Analogy's source code
    def l2_regularizer(self, batch, reg):
        e1 = self.entity_emb(batch[:, 0])
        r = self.relation_emb(batch[:, 1])
        e2 = self.entity_emb(batch[:, 2])
        return reg / (2 * e1.size(-1)) * e1.pow(2).sum() \
               + reg / (2 * e2.size(-1)) * e2.pow(2).sum() \
               + reg / (2 * r.size(-1)) * r.pow(2).sum()

    def lifted_constraints(self, batch, reg, delta):
        if reg == 0:
            return 0
        out = 0
        r = batch[:, 1]
        f = nn.ReLU()
        for head in self.rules.keys():
            n = np.count_nonzero(r == head)
            if n == 0:
                continue
            for body in self.rules[head]:
                out += n * torch.dot(f((self.relation_emb(torch.LongTensor([body]))
                                        - self.relation_emb(torch.LongTensor([head]))) + delta)[0],
                                     torch.ones(self.relation_emb(torch.LongTensor([head])).size(-1)))
        return reg * out


from models.Analogy import *
from models.DistMult import *
from models.Complex import *
from models.ConvE import *
from models.TransE import *
