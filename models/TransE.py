from models.base_model import BaseModel
import torch
from config import Config

class TransE(BaseModel):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(TransE, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)
        self.normalize_init_all(self.entity_emb.weight.data)
        self.normalize_init_all(self.relation_emb.weight.data)
        self.last_batch = None
        self.E1_R = None
        self.R_E2 = None
        # special settings for TransE
        self.num_negatives = None
        self.norm = None
        self.L1 = None
        self.fromConfig()


    # pull special settings required for TransE from config file
    def fromConfig(self):
        self.num_negatives = Config.num_negatives
        self.norm = Config.norm  #eval(Config.norm)
        self.L1 = Config.L1  #eval(Config.L1)


    def forward(self, x):
        if self.last_batch is not None:
            previous_embs = list(set(self.last_batch[:, 0]).union(set(self.last_batch[:, 2])))
            self.normalize_ents(previous_embs)
        self.last_batch = x

        e1 = self.entity_emb(x[:, 0])
        r = self.relation_emb(x[:, 1])
        e2 = self.entity_emb(x[:, 2])
        return self.forward_emb(e1, r, e2)

    def normalize_ents(self, indices):
        # normalize only embeddings affected by previous batch for speed
        num_entity = len(indices)
        entities = self.entity_emb.weight.data[indices]
        entity_norm = torch.norm(entities, p=2, dim=1)
        entity_norm_matrix = entity_norm.view(num_entity, 1).expand_as(entities)
        self.entity_emb.weight.data[indices] = entities.div(entity_norm_matrix)  # update

    def normalize_init_all(self, weights):
        num = weights.shape[0]
        norm = torch.norm(weights, p=2, dim=1)
        norm_matrix = norm.view(num,1).expand_as(weights)
        weights = weights.div(norm_matrix)  # update


    def forward_emb(self, e1, r, e2):
        n = int(e1.shape[0])  # batch size
        n_scores = self.num_negatives + 1
        n_pos = int(n / n_scores)
        n_neg = n - n_pos

        if self.num_negatives > 1:
            # upsample positive tuples
            tmp_e1 = torch.cat((torch.cat([e1[0:n_pos]] * (self.num_negatives - 1)), e1), 0)
            tmp_r = torch.cat((torch.cat([r[0:n_pos]] * (self.num_negatives - 1)), r), 0)
            tmp_e2 = torch.cat((torch.cat([e2[0:n_pos]] * (self.num_negatives - 1)), e2), 0)

            # split into positive and corrupted tuples
            tmp_e1 = tmp_e1.view(2, n_neg, e1.shape[1])
            tmp_r = tmp_r.view(2, n_neg, r.shape[1])
            tmp_e2 = tmp_e2.view(2, n_neg, e2.shape[1])

        else:
            # just split into positive and corrupted tuples
            tmp_e1 = e1.view(2, n_neg, e1.shape[1])
            tmp_r = r.view(2, n_neg, r.shape[1])
            tmp_e2 = e2.view(2, n_neg, e2.shape[1])

        # prepare for nn.MaxMarginLoss
        out = (tmp_e1 + tmp_r) - tmp_e2

        if self.L1:  # L1 norm dissimilarity
            out = torch.sum(torch.abs(out), 2)

        else:  # L2 norm dissimilarity
            out = torch.sqrt(torch.sum(out.pow(2), 2))

        labels = torch.ones((out[0].shape[0],)) * -1

        return out[0], out[1], labels


    def precompute(self, r):
        r_emb = self.relation_emb(r)
        r_matrix = r_emb.repeat(self.num_entities, 1)
        self.R_E2 = r_matrix - self.entity_weights
        self.E1_R = self.entity_weights + r_matrix

    def scores_e2(self, E1):
        E1_embs = self.entity_emb(E1)#.long()
        E1_R_E2 = []
        for e1 in E1_embs:
            e1 = e1.repeat(self.num_entities, 1)
            e1_r_e2 =  e1 + self.R_E2
            E1_R_E2.append(e1_r_e2)
        E1_R_E2 = torch.stack(E1_R_E2)
        dim = 1 if len(E1) >= 2 else 0
        if self.norm == 0:
            return -torch.sum(E1_R_E2**2, dim)
        else:
            return -torch.sum(torch.abs(E1_R_E2), dim)

    def scores_e1(self, E2):
        E2_embs = self.entity_emb(E2)
        E1_R_E2 = []
        for e2 in E2_embs:
            e2 = e2.repeat(self.num_entities, 1)
            e1_r_e2 = self.E1_R - e2
            E1_R_E2.append(e1_r_e2)
        E1_R_E2 = torch.stack(E1_R_E2)
        dim = 1 if len(E2) >= 2 else 0
        if self.norm == 0:
            return -torch.sum(E1_R_E2**2, dim)
        else:
            return -torch.sum(torch.abs(E1_R_E2), dim)

    # To implement
    '''
    def score_matrix_r(self, r):
        pass
    '''
