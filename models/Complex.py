from models.base_model import BaseModel
import torch

class Complex(BaseModel):
    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(Complex, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)
        self.E1_R = None
        self.R_E2 = None

    def forward_emb(self, e1, r, e2):
        p = int(e1.shape[0])  # batch size
        q = int(e1.shape[1] / 2)  # half of dim
        # A.shape = [p, q]; B.shape = [p, q]
        # batch-wise dot product: torch.bmm(A,view(p, 1, q), B.view(p, q, 1))
        out = torch.bmm((self.ent_func(e1[:, :q]) * r[:, :q]).view(p, 1, q), self.ent_func(e2[:, :q]).view(p, q, 1))
        out += torch.bmm((self.ent_func(e1[:, :q]) * r[:, q:]).view(p, 1, q), self.ent_func(e2[:, q:]).view(p, q, 1))
        out += torch.bmm((self.ent_func(e1[:, q:]) * r[:, :q]).view(p, 1, q), self.ent_func(e2[:, q:]).view(p, q, 1))
        out -= torch.bmm((self.ent_func(e1[:, q:]) * r[:, q:]).view(p, 1, q), self.ent_func(e2[:, :q]).view(p, q, 1))
        out = out[:, 0, 0]
        return out

    def create_matrix(self, r):
        r_emb = self.relation_emb(r)
        half = int(len(self.entity_weights[0]) / 2)
        diag1 = r_emb[:half]
        diag2 = r_emb[half:]
        mat1 = torch.diagflat(diag1)
        mat2 = torch.diagflat(diag2)
        mat3 = -mat2
        R = torch.cat( (torch.cat( (mat1, mat2), 1), torch.cat( (mat3, mat1), 1)), 0)
        return R

    def precompute(self, r):
        R = self.create_matrix(r)
        self.E1_R = torch.matmul(self.entity_weights, R)
        self.R_E2 = torch.matmul(R, torch.transpose(self.entity_weights,0,1))


    def score_matrix_r(self, r):
        R = self.create_matrix(r)
        R_E2 = torch.matmul(R, torch.transpose(self.entity_weights,0,1))
        return torch.matmul(self.entity_weights, R_E2)


    def scores_e2(self, E1):
        e1_embs = self.entity_emb(E1) # lookup embedding vectors for e1 entries: n x dim
        return torch.matmul(e1_embs, self.R_E2)


    def scores_e1(self, E2):
        e2_embs = self.entity_emb(E2)

        # if there is only one triple for a relation -> do not transpose
        if len(e2_embs.shape) >= 2:
            e2_embs = torch.transpose(e2_embs, 0, 1)
        score = torch.matmul(self.E1_R, e2_embs)

        if len(e2_embs.shape) >= 2:
            score = torch.transpose(score, 0, 1)
        return score


