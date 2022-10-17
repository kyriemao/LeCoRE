import torch
import torch.nn as nn
from IPython import embed


def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)

def cal_mae_loss(query_embs, oracle_query_embs):
    loss_func = nn.L1Loss(reduction="none")
    mae_loss = loss_func(query_embs, oracle_query_embs)
    return torch.mean(torch.sum(mae_loss, dim=1))
    
def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    if neg_doc_embs is not None:
        neg_ratio = int(neg_doc_embs.shape[0] / query_embs.shape[0])
        neg_scores = torch.sum(query_embs.unsqueeze(1) * neg_doc_embs.view(batch_size, neg_ratio, -1), dim = -1) # B * neg_ratio
        score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + neg_ratio)  in_batch negatives + neg_ratio other negatives
    label_mat = torch.arange(batch_size).to(query_embs.device)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss


# Regularizations
class L1:

    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


class SparsityRatio:
    """non-differentiable
    """

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def __call__(self, batch_rep):
        return 1 - torch.sum(batch_rep != 0, dim=-1).float().mean() / self.output_dim


def init_regularizer(reg, **kwargs):
    if reg == "None":
        return None
    elif reg == "L0":
        return L0()
    elif reg == "sparsity_ratio":
        return SparsityRatio(output_dim=kwargs["output_dim"])
    elif reg == "L1":
        return L1()
    elif reg == "FLOPS":
        return FLOPS()
    else:
        raise NotImplementedError("provide valid regularizer")
