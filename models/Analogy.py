from models.base_model import BaseModel
import torch

class Analogy(BaseModel):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(Analogy, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)
        self.num_scalars = entity_emb_size/2
        self.E1_R = None
        self.R_E2 = None

    def forward_emb(self, e1, r, e2):
        p = int(e1.shape[0])  # batch size
        n = int(self.num_scalars)
        q = int(n + n / 2)

        out1 = torch.sum(self.ent_func(e1[:, :n]) * r[:, :n] * self.ent_func(e2[:, :n]), 1)

        out2 = torch.bmm((self.ent_func(e1[:, n:q]) * r[:, n:q]).view(p, 1, int(n / 2)),
                         self.ent_func(e2[:, n:q]).view(p, int(n / 2), 1))
        out2 += torch.bmm((self.ent_func(e1[:, n:q]) * r[:, q:]).view(p, 1, int(n / 2)),
                          self.ent_func(e2[:, q:]).view(p, int(n / 2), 1))
        out2 += torch.bmm((self.ent_func(e1[:, q:]) * r[:, n:q]).view(p, 1, int(n / 2)),
                          self.ent_func(e2[:, q:]).view(p, int(n / 2), 1))
        out2 -= torch.bmm((self.ent_func(e1[:, q:]) * r[:, q:]).view(p, 1, int(n / 2)),
                          self.ent_func(e2[:, n:q]).view(p, int(n / 2), 1))
        out2 = out2[:, 0, 0]

        return out1 + out2

    def create_matrix(self, r):
        r_emb = self.relation_emb(r)
        diag1 = r_emb[:int(self.num_scalars)]
        comp1 = torch.diagflat(diag1)
        block_diag = r_emb[int(self.num_scalars):]
        half = int(len(block_diag) / 2)

        diag2 = block_diag[:half]
        diag3 = block_diag[half:]

        mat2_1 = torch.diagflat(diag2)
        mat2_2 = torch.diagflat(diag3)
        mat2_3 = -mat2_2

        # comp2: create following block:
        # [mat2_1, mat2_2],
        # [mat2_3, mat2_1]
        comp2 = torch.cat( (torch.cat( (mat2_1, mat2_2), 1), torch.cat((mat2_3, mat2_1), 1)), 0)
        R = torch.cat( (torch.cat((comp1, torch.zeros_like(comp1)), 1) , torch.cat((torch.zeros_like(comp1), comp2), 1)), 0)
        return R


    def score_matrix_r(self, r):
        R = self.create_matrix(r)
        R_E2 = torch.matmul(R, torch.transpose(self.entity_weights,0,1))
        return torch.matmul(self.entity_weights, R_E2)

    def precompute(self, r):
        R = self.create_matrix(r)
        self.E1_R = torch.matmul(self.entity_weights, R)
        self.R_E2 = torch.matmul(R, torch.transpose(self.entity_weights,0,1))


    def scores_e2(self, E1):
        e1_embs = self.entity_emb(E1) # lookup embedding vectors for e1 entries: n x dim
        return torch.matmul(e1_embs, self.R_E2)   # or: e1_embs @ R_E2


    def scores_e1(self, E2):
        e2_embs = self.entity_emb(E2)
        # if there is only one triple for a relation -> do not transpose
        if len(e2_embs.shape) >= 2:
            e2_embs = torch.transpose(e2_embs, 0, 1)
        score = torch.matmul(self.E1_R, e2_embs)

        if len(e2_embs.shape) >= 2:
            score = torch.transpose(score, 0, 1)
        return score    # n x num_entities
