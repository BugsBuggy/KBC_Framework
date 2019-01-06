import torch
from models.base_model import BaseModel


class DistMult(BaseModel):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(DistMult, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)
        self.E1_R = None
        self.R_E2 = None


    def forward_emb(self, e1, r, e2):
        return torch.sum(self.ent_func(e1) * r * self.ent_func(e2), 1)


    def precompute(self, r):
        R = self.relation_emb(r)
        R_diag = torch.diagflat(R)
        self.E1_R = torch.matmul(self.entity_weights, R_diag)
        self.R_E2 = torch.matmul(R_diag, torch.transpose(self.entity_weights,0,1))

    def scores_e2(self, E1):
        e1_embs = self.entity_emb(E1)
        return torch.matmul(e1_embs, self.R_E2)

    def scores_e1(self, E2):
        e2_embs = self.entity_emb(E2)

        # if there are multiple triples for one relation -> transpose
        if len(e2_embs.shape) >= 2:
            e2_embs = torch.transpose(e2_embs, 0, 1)
        score = torch.matmul(self.E1_R, e2_embs)

        if len(e2_embs.shape) >= 2:
            score = torch.transpose(score, 0, 1)
        return score

    # should expect a tensor
    def score_matrix_r(self, r):
        R = self.relation_emb(r)
        R_diag = torch.diagflat(R)

        E1_R = torch.matmul(self.entity_weights, R_diag)
        E2 = self.entity_weights
        if len(self.entity_weights.shape) >= 2:
            E2 = torch.transpose(E2, 0, 1)
        return torch.matmul(E1_R, E2)