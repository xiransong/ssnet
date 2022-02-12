import torch
from torch import nn
import math


class EmbNorm(torch.nn.Module):
    
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, embs):
        return embs / torch.norm(embs, p=self.p, dim=-1).view(-1, 1)


class ConcatEmbsPairsOperator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embs_pairs):
        feats = embs_pairs.reshape(len(embs_pairs), -1)
        return feats
    

class HadamardEmbsPairsOperator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embs_pairs):
        emb0 = embs_pairs[:, 0, :]
        emb1 = embs_pairs[:, 1, :]
        feats = emb0 * emb1
        return feats


class DotProductPairsOperator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embs_pairs):
        return (embs_pairs[:,0,:] * embs_pairs[:,1,:]).sum(-1).unsqueeze(-1)


class CatHadPairsOperator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cat = ConcatEmbsPairsOperator()
        self.had = HadamardEmbsPairsOperator()
    
    def forward(self, embs_pairs):
        return torch.cat((self.cat(embs_pairs), self.had(embs_pairs)), dim=-1)


class TransformerLayer(nn.Module):

    def __init__(self, dim_input, rows_input, qk_dim):
        super(TransformerLayer, self).__init__()

        v_dim = dim_input
        self.Att1 = Attention(dim_input, qk_dim, v_dim)
        self.LN11 = LayerNorm(input_size=(rows_input, dim_input))

        self.MLP1 = MLP(d_model=v_dim)
        self.LN12 = LayerNorm(input_size=(rows_input, dim_input))

        self.Att2 = Attention(dim_input, qk_dim, v_dim)
        self.LN21 = LayerNorm(input_size=(rows_input, dim_input))

        self.MLP2 = MLP(d_model=v_dim)
        self.LN22 = LayerNorm(input_size=(rows_input, dim_input))

    def forward(self, X):
        Y = self.Att1(X)
        X = self.LN11(X + Y)
        Y = self.MLP1(X)
        X = self.LN12(X + Y)

        Y = self.Att2(X)
        X = self.LN21(X + Y)
        Y = self.MLP2(X)
        X = self.LN22(X + Y)

        return X


class Attention(nn.Module):

    def __init__(self, input_dim, qk_dim, v_dim, out_fn=None):
        super(Attention, self).__init__()
        self.out_fn = out_fn
        self.coe = 1 / math.sqrt(qk_dim)

        Wq = torch.FloatTensor(input_dim, qk_dim)
        nn.init.xavier_uniform_(Wq)
        self.Wq = nn.Parameter(Wq)

        Wk = torch.FloatTensor(input_dim, qk_dim)
        nn.init.xavier_uniform_(Wk)
        self.Wk = nn.Parameter(Wk)

        Wv = torch.FloatTensor(input_dim, v_dim)
        nn.init.xavier_uniform_(Wv)
        self.Wv = nn.Parameter(Wv)

    def forward(self, X):
        Q = torch.matmul(X, self.Wq)
        K = torch.matmul(X, self.Wk)
        V = torch.matmul(X, self.Wv)
        Wa = torch.softmax(self.coe * torch.matmul(Q, K.transpose(-1, -2)), dim=-1)
        out = torch.matmul(Wa, V)
        if self.out_fn is not None:
            return self.out_fn(out)
        return out


class LayerNorm(nn.Module):

    def __init__(self, input_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(input_size))
        self.b_2 = nn.Parameter(torch.zeros(input_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.05):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
