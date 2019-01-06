from models.base_model import  BaseModel
from config import Config
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

class ConvE(BaseModel):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(ConvE, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)
        self.r = None
        self.model = None

        # BatchNormalization
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm1d(entity_emb_size)

        # Dropout
        self.drop1 = nn.Dropout(Config.drop1)
        self.drop2 = nn.Dropout(Config.drop2)
        self.drop_channel = nn.Dropout2d(Config.drop_channel)

        # Convolution
        self.conv = nn.Conv2d(1, 32, (3, 3), 1, 0)

        # Linear Projection
        self.lp = nn.Linear(10368, entity_emb_size)

    def forward_emb(self, e1, r, e2):
        n = int(e1.shape[0])  # batch size
        dim = int(e1.shape[1])  # embedding dimension

        e1_shaped = e1.view(-1, 1, 10, 20)
        r_shaped = r.view(-1, 1, 10, 20)

        # Input Stabilization
        out = torch.cat([e1_shaped, r_shaped], 2)
        out = self.bn1(out)
        out = self.drop1(out)

        # Convolution + Stabilization
        out = self.conv(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.drop_channel(out)

        # Linear Projection + Stabilization
        out = out.view(n, -1)
        out = self.lp(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.drop2(out)

        # Batch-wise Inner product
        out = torch.bmm(out.view(n, 1, dim), e2.view(n, dim, 1))
        out = out[:, 0, 0]

        return out

    # store r for scores_e2, scores_e2
    def precompute(self, r):
        self.r = r

    ''' finish this
    def scores_e1(self, E2):
        num_entities = self.num_entities.item()
        zeros = torch.zeros([num_entities]).long()
        r_column = torch.zeros([num_entities]).long()
        r_column = r_column.fill_(self.r)
        all_entities = torch.arange(0, num_entities).long()
        input = torch.transpose(torch.stack( (all_entities, r_column, zeros) ), 0, 1)
        scores_e2 = []

        for e2 in E2:
            input[:, 2].fill_(e2)
            score = self.model(input)
            scores_e2.append(score)

        return torch.stack(scores_e2)

    def scores_e2(self, E1):
        num_entities = self.num_entities.item()
        zeros = torch.zeros([num_entities]).long()
        r_column = torch.zeros([num_entities]).long()
        r_column = r_column.fill_(self.r)
        all_entities = torch.arange(0, num_entities).long()

        input = torch.transpose(torch.stack( (zeros, r_column, all_entities) ), 0, 1)

        scores_e2 = []
        for e1 in E1:
            input[:, 0].fill_(e1)
            score = self.model(input)
            scores_e2.append(score)

        scores_e2 = torch.stack(scores_e2)
        return scores_e2
    '''
