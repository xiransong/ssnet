import Module.gnn as gnn_module
from Module.gnn import TwoLayerMLP
from Module.gnn import LightGCN, SAGE, GAT, GIN, FAGCN
from Wrapper.base_emb_table import EmbTable
from Module.lossfunction import bpr_loss
from Utils import io
from Utils.metric import all_metrics

import numpy as np
import numba
import torch
import dgl
import os.path as osp
from copy import deepcopy
from tqdm import tqdm


class DegreeDiscriminator(torch.nn.Module):
    
    def __init__(self, all_degrees):
        super().__init__()
        _degrees = all_degrees.numpy()
        self.p25 = np.percentile(_degrees, 25)
        self.p50 = np.percentile(_degrees, 50)
        self.p75 = np.percentile(_degrees, 75)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
        )
        
    def forward(self, embs, degrees):
        label0 = degrees < self.p25
        label1 = (self.p25 <= degrees) & (degrees < self.p50)
        label2 = (self.p50 <= degrees) & (degrees < self.p75)
        label3 = degrees >= self.p75
        
        pred_logits = self.mlp(embs)
        true_logits = torch.cat((
            label0.reshape(-1,1).float(), 
            label1.reshape(-1,1).float(), 
            label2.reshape(-1,1).float(), 
            label3.reshape(-1,1).float()
        ), dim=-1)
        
        return pred_logits, true_logits.to(pred_logits.device)


@numba.jit(nopython=True)
def degree_mapping(degrees):
    h_idx = np.zeros(len(degrees))
    for i in range(len(degrees)):
        d = degrees[i]
        if d > 99:
            h_idx[i] = 19
        else:
            h_idx[i] = d // 5
    return h_idx  


class MLP(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['scaling']:
            print("## scaling")
            
            if self.config['cat_self']:
                print("### cat_self")
                self.mlp_scalar = torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 1, bias=False),
                    # torch.nn.Sigmoid()
                )
            else:
                if '1layer-mlp_scaling' in config['str_set']:
                    print("## 1layer-mlp_scaling")
                    self.mlp_scalar = torch.nn.Sequential(
                        torch.nn.Linear(64, 1, bias=False),
                        # torch.nn.Sigmoid()
                    )
                elif '3layer-mlp_scaling' in config['str_set']:
                    print("## 3layer-mlp_scaling")
                    self.mlp_scalar = torch.nn.Sequential(
                        torch.nn.Linear(64, 48),
                        torch.nn.Tanh(),
                        torch.nn.Linear(48, 32),
                        torch.nn.Tanh(),
                        torch.nn.Linear(32, 1, bias=False),
                        # torch.nn.Sigmoid()
                    )
                else:  # 2layer
                    print("## 2layer-mlp_scaling")
                    self.mlp_scalar = torch.nn.Sequential(
                        torch.nn.Linear(64, 32),
                        torch.nn.Tanh(),
                        torch.nn.Linear(32, 1, bias=False),
                        # torch.nn.Sigmoid()
                    )
            self.sigmoid = torch.nn.Sigmoid()
            if 'tao' in self.config:
                self.tao = self.config['tao']
            else:
                self.tao = 11.0
            
        elif config['l2norm']:
            print("## l2norm")
        elif config['ffn'] or config['res_ffn']:
            if config['fnn']:
                print("## fnn")
            else:
                print("## res_ffn")
            if config['linear']:
                print("### linear")
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                )
                with torch.no_grad():
                    self.mlp[0].weight[:] = torch.eye(64)
                    self.mlp[0].bias[:] = torch.zeros(64)
            else:
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                )
        else:
            assert 0
        
    def forward(self, embs, self_emb=None, return_scalars=False):
        if self.config['scaling']:
            if self_emb is not None:
                w = self.mlp_scalar(torch.cat((embs, self_emb), dim=-1))
            else:
                w = self.mlp_scalar(embs)
            if self.config['use_tao']:
                w /= self.tao
            embs = embs * self.sigmoid(w)
            
            if return_scalars:
                return embs, w
            else:
                return embs
        elif self.config['ffn']:
            return self.mlp(embs)
        elif self.config['res_ffn']:
            return self.mlp(embs) + embs
        else:  # l2norm
            return embs / torch.norm(embs, p=2, dim=-1, keepdim=True)
    
    def update_tao(self):
        if self.tao > 1.1:
            self.tao -= 1
        print("## tao = ", self.tao)


class LearnableScalar(torch.nn.Module):
    
    def __init__(self, num_nodes, config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(num_nodes).reshape(-1, 1))
        self.mlp = None
        if 'learnable_scalar_plus_mlp_scaling' in config['str_set']:
            print("## learnable_scalar_plus_mlp_scaling")
            self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(64, 32),
                    torch.nn.Tanh(),
                    torch.nn.Linear(32, 1, bias=False),
                    torch.nn.Sigmoid()
            )
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, embs, idx=None):
        if idx is None:
            _w = self.sigmoid(self.w)
        else:
            _w = self.sigmoid(self.w[idx])
        
        if self.mlp is not None:
            return embs * (_w + self.mlp(embs))
        else:
            return embs * _w


class DegreeMLP(torch.nn.Module):
    
    def __init__(self, degrees):
        super().__init__()
        self.h_idx = degree_mapping(degrees)
        self.hot = torch.eye(20)
        for i in range(1, len(self.hot)):
            self.hot[i][i-1] = 1
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64 + 20, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64)
        )
        
    def forward(self, embs, idx=None):
        if idx is None:
            H = self.hot[self.h_idx]
        else:
            H = self.hot[self.h_idx[idx]]
        return self.mlp(torch.cat((embs, H.to(embs.device)), dim=-1))


class BaseEmbeddingModel:
    
    def __init__(self):
        pass
    
    def __call__(self, batch_data):
        return self.forward(batch_data)
    
    def parameters(self):
        return self.opt_param_list
    
    def val_or_test_a_batch(self, batch_data):
        batch_results = self.inference(batch_data, 
                                    eval_on_whole_graph=False)
        return batch_results
    
    def load(self, root):
        config_file = osp.join(root, "config.yaml")
        if osp.exists(config_file):
            self.config = io.load_yaml(config_file)
        self.out_emb_table = torch.load(osp.join(root, "out_emb_table.pt"))
        
    def inference(self, batch_data, eval_on_whole_graph=False, 
                  only_return_pos_score=False,
                  only_return_pos_neg_score=False):
        src, dst, neg_dst = batch_data
        
        src_emb = self.out_emb_table[src]
        dst_emb = self.out_emb_table[dst]

        if eval_on_whole_graph:
            pos_score = torch.sum(src_emb * dst_emb, dim=-1)
            neg_score = src_emb @ self.out_emb_table.t()
        else:
            neg_dst_emb = self.out_emb_table[neg_dst]
            pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, dst_emb, neg_dst_emb)
        
        pos_neg_score = torch.cat((pos_score.view(-1, 1), neg_score), dim=-1).detach().numpy()
        batch_results = all_metrics(pos_neg_score)
        return batch_results


class BaseGNNModel(BaseEmbeddingModel):

    def __init__(self):
        pass
    
    def build_basic(self, data, config):
        self.config = config
        self.device = config['device']
        
        if 'node_collator' in data:
            self.node_collator = data['node_collator']  # for inference
        
        self.base_emb_table = EmbTable()
        self.base_emb_table.build(data, config)
        self.out_emb_table = None
        
        self.opt_param_list = {'dense':[], 'sparse':[]}
        self.opt_param_list['dense'].extend(self.base_emb_table.parameters()['dense'])
        self.opt_param_list['sparse'].extend(self.base_emb_table.parameters()['sparse'])
    
    def build_gnn_and_mlp(self, GNN, kwargs):
        # in: self.config, self.device, self.opt_param_list
        # out: self.gnn, self.mlp/self.mlp_list
        if self.config['final_layer_mlp']:
            self.mlp = MLP(self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.mlp.parameters(), 'lr': 0.001})
            self.gnn = GNN(**kwargs)
        
        # elif self.config['each_layer_mlp']:
        #     self.mlp_list = torch.nn.ModuleList([
        #         MLP(self.config).to(self.device) for _ in range(1 + self.config['num_gcn_layer'])
        #     ])
        #     # self.opt_param_list['dense'].append({'params': self.mlp_list.parameters(), 'lr': 0.001})  # these params belong to gnn
        #     self.gnn = GNN(**kwargs, mlp_list=self.mlp_list)
            
        else:
            self.gnn = GNN(**kwargs)
            
        self.gnn = self.gnn.to(self.device)
        self.opt_param_list['dense'].append({'params': self.gnn.parameters(), 'lr': self.config['gnn_lr']})
    
    def build(self, data, config):
        self.build_basic(data, config)
        # init: self.gnn, self.mlp
        pass
    
    def forward(self, batch_data):
        batch_nids, local_idx, input_nids, output_nids, blocks = batch_data
    
        blocks = [block.to(self.device) for block in blocks]
        
        output_embs = self.gnn(
            blocks, self.base_emb_table[input_nids]
        )
        
        output_embs = output_embs[local_idx].view(3, -1, self.base_emb_table.shape[-1])
        
        src_emb = output_embs[0, :, :]
        pos_emb = output_embs[1, :, :]
        neg_emb = output_embs[2, :, :]
        
        if self.config['final_layer_mlp']:
            src_emb = self.mlp(src_emb)
            pos_emb = self.mlp(pos_emb)
            neg_emb = self.mlp(neg_emb)
        
        pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, pos_emb, neg_emb)
        loss = bpr_loss(pos_score, neg_score)
        
        if self.config['gnn_reg']:
            reg_loss = 0
            for param in self.gnn.parameters():
                reg_loss += torch.norm(param)
            loss += self.config['gnn_reg_weight'] * reg_loss
        
        return loss
    
    def prepare_for_train(self):
        del self.out_emb_table
      
    def prepare_for_val(self):
        self.base_emb_table.del_grad_matrix()
        self.out_emb_table = torch.empty(self.base_emb_table.shape, dtype=torch.float32)
        dl = torch.utils.data.DataLoader(dataset=torch.arange(len(self.base_emb_table)), 
                                         batch_size=10240,
                                         collate_fn=self.node_collator.collate)
        
        for input_nids, output_nids, blocks in tqdm(dl, desc="get all gnn output embs"):
            blocks = [block.to(self.device) for block in blocks]
            output_embs = self.gnn(
                blocks, self.base_emb_table[input_nids]
            )
            self.out_emb_table[output_nids] = output_embs.cpu()
        
        if self.config['final_layer_mlp']:
            _mlp = deepcopy(self.mlp).cpu()
            if self.config['scaling']:
                self.out_emb_table, self.mlp_output_scalars = _mlp(self.out_emb_table, return_scalars=True)
            else:
                self.out_emb_table = _mlp(self.out_emb_table, return_scalars=False)
    
    def prepare_for_test(self):
        self.base_emb_table.del_grad_matrix()
    
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
        torch.save(self.gnn.state_dict(), osp.join(root, "gnn.pt"))
        torch.save(self.base_emb_table.get_full_matrix().cpu(), osp.join(root, "base_emb_table.pt"))
        if self.config['final_layer_mlp']:
            if not self.config['l2norm']:
                torch.save(self.mlp, osp.join(root, "mlp.pt"))
            if self.config['scaling']:
                torch.save(self.mlp_output_scalars, osp.join(root, "mlp_output_scalars.pt"))
        if self.config['each_layer_mlp']:
            torch.save(self.mlp_list, osp.join(root, "mlp_list.pt"))


class LightGCNWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)

        self.full_adj = data['full_adj']
        self.lightgcn = LightGCN()

        if self.config['final_layer_mlp']:
            self.mlp = MLP(self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.mlp.parameters(), 'lr': 0.001})
            
        # elif self.config['each_layer_mlp']:
        #     self.mlp_list = torch.nn.ModuleList([
        #         MLP(self.config).to(self.device) for _ in range(1 + self.config['num_gcn_layer'])
        #     ])
        #     self.opt_param_list['dense'].append({'params': self.mlp_list.parameters(), 'lr': 0.001})
        
        if self.config['learnable_scalar']:
            self.learnable_scalar = LearnableScalar(num_nodes=len(self.base_emb_table), config=self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.learnable_scalar.parameters(), 'lr': 0.001})

        if self.config['degree_mlp']:
            g = io.load_pickle(osp.join(self.config['data_root'], "train_undi_graph.np_csr.pkl"))
            degrees = g['ptr_array'][:, 1]
            self.degree_mlp = DegreeMLP(degrees).to(self.device)
            self.opt_param_list['dense'].append({'params': self.degree_mlp.parameters(), 'lr': 0.001})
        
        self.in_degrees = data['in_degrees']
        
        ## degree discriminator
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.degree_discriminator = DegreeDiscriminator(self.in_degrees).to(self.device)
            self.degree_discriminator_opt = torch.optim.Adam(
                self.degree_discriminator.parameters(), lr=0.001
            )
       
    def forward(self, batch_data):
        if self.config['num_gcn_layer'] == 2:
            batch_nids, local_idx, B1, B2 = batch_data
            input_nids, output_nids, blocks = B1
            
            # calc layer1
            blocks = [block.to(self.device) for block in blocks]
            output_embs = self.lightgcn(
                blocks, self.base_emb_table[input_nids]
            )
            X1 = output_embs[local_idx]
            
            # calc layer2
            input_nids, output_nids, blocks = B2
            blocks = [block.to(self.device) for block in blocks]
            output_embs = self.lightgcn(
                blocks, self.base_emb_table[input_nids]
            )
            X2 = output_embs[local_idx]
            
            # final output emb
            output_embs = X2
        
        else:
            batch_nids, local_idx, input_nids, output_nids, blocks = batch_data
        
            # calc layer1
            blocks = [block.to(self.device) for block in blocks]
            
            X0_in = self.base_emb_table[input_nids]
                
            output_embs = self.lightgcn(
                blocks, X0_in
            )
            X1 = output_embs[local_idx]
            
            output_embs = X1
        
        
        if self.config['final_layer_mlp']:
            output_embs = self.mlp(output_embs)
        
        if self.config['learnable_scalar']:
            output_embs = self.learnable_scalar(output_embs, batch_nids)

        if self.config['degree_mlp']:
            output_embs = self.degree_mlp(output_embs, batch_nids)
        
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            batch_embs = output_embs
            nids = batch_nids
            degrees = self.in_degrees[nids]
            
            self.degree_discriminator_opt.zero_grad()
            
            pred_logits, true_logits = self.degree_discriminator(
                batch_embs.detach(), degrees)
            discriminator_loss = self.bce_loss(pred_logits, true_logits)
            
            discriminator_loss.backward()
            self.degree_discriminator_opt.step()
            
            # _degree_discriminator = self.degree_discriminator.detach()
            pred_logits, true_logits = self.degree_discriminator(
                batch_embs, degrees)
            _discriminator_loss = self.bce_loss(pred_logits, true_logits)
        
        output_embs = output_embs.view(3, -1, self.base_emb_table.shape[-1])
        
        src_emb = output_embs[0, :, :]
        pos_emb = output_embs[1, :, :]
        neg_emb = output_embs[2, :, :]
        
        pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, pos_emb, neg_emb)
        loss = bpr_loss(pos_score, neg_score)
        
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            return loss - self.config['gan_gen_loss_weight'] * _discriminator_loss
        else:
            return loss
    
    def prepare_for_val(self):
        self.base_emb_table.del_grad_matrix()
        X = self.base_emb_table.get_full_matrix().cpu()
        self.out_emb_table = torch.sparse.mm(self.full_adj, X)
        if self.config['num_gcn_layer'] == 2:
            self.out_emb_table = torch.sparse.mm(self.full_adj, self.out_emb_table)
        
        if self.config['final_layer_mlp']:
            _mlp = deepcopy(self.mlp).cpu()
            if self.config['scaling']:
                self.out_emb_table, self.mlp_output_scalars = _mlp(self.out_emb_table, return_scalars=True)
            else:
                self.out_emb_table = _mlp(self.out_emb_table, return_scalars=False)
        
        if self.config['learnable_scalar']:
            _learnable_scalar = deepcopy(self.learnable_scalar).cpu()
            self.out_emb_table = _learnable_scalar(self.out_emb_table)
            
        if self.config['degree_mlp']:
            _degree_mlp = deepcopy(self.degree_mlp).cpu()
            self.out_emb_table = _degree_mlp(self.out_emb_table)
            
        if self.config['gan_end_to_end']:
            _degree_discriminator = deepcopy(self.degree_discriminator).cpu()
            pred_logits, true_logits = _degree_discriminator(
                self.out_emb_table, self.in_degrees
            )
            pred_label = pred_logits.argmax(dim=-1)
            true_label = true_logits.argmax(dim=-1)
            precision = (pred_label == true_label).float().mean().item()
            print("## degree_discriminator precision:", precision)
            
            file_gan = osp.join(self.config['results_root'], 'gan_record.txt')
            
            if not osp.exists(file_gan):
                with open(file_gan, "w") as f:
                    f.write('{:.6g}\n'.format(precision))
            else:    
                with open(file_gan, "a") as f:
                    f.write('{:.6g}\n'.format(precision))
        
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
        torch.save(self.base_emb_table.get_full_matrix().cpu(), osp.join(root, "base_emb_table.pt"))
        if self.config['final_layer_mlp']:
            if not self.config['l2norm']:
                torch.save(self.mlp, osp.join(root, "mlp.pt"))
            if self.config['scaling']:
                torch.save(self.mlp_output_scalars, osp.join(root, "mlp_output_scalars.pt"))
        if self.config['learnable_scalar']:
            torch.save(self.learnable_scalar.w.data.cpu(), osp.join(root, "learnable_scalar.pt"))
        if self.config['degree_mlp']:
            torch.save(self.degree_mlp, osp.join(root, "degree_mlp.pt"))
        
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            torch.save(self.degree_discriminator, osp.join(root, "degree_discriminator.pt"))


class PPRGoWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.config = config
        self.device = config['device']
        
        self.base_emb_table = EmbTable()
        self.base_emb_table.build(data, config)
        self.out_emb_table = None
        
        self.opt_param_list = {'dense':[], 'sparse':[]}
        self.opt_param_list['dense'].extend(self.base_emb_table.parameters()['dense'])
        self.opt_param_list['sparse'].extend(self.base_emb_table.parameters()['sparse'])
        
        # try:
        #     raw_nei = io.load_pkl_big_np_array(osp.join(config['ppr_data_root'], "appr_nei_array.pkl"))
        #     raw_wei = io.load_pkl_big_np_array(osp.join(config['ppr_data_root'], "appr_sym_normalized_wei_array.pkl"))
        # except EOFError:
        #     raw_nei = io.load_pickle(osp.join(config['ppr_data_root'], "appr_nei_array.pkl"))
        #     raw_wei = io.load_pickle(osp.join(config['ppr_data_root'], "appr_sym_normalized_wei_array.pkl"))
        
        try:
            raw_nei = io.load_pkl_big_np_array(osp.join(config['ppr_data_root'], "nei.pkl"))
            raw_wei = io.load_pkl_big_np_array(osp.join(config['ppr_data_root'], "wei.pkl"))
        except EOFError:
            raw_nei = io.load_pickle(osp.join(config['ppr_data_root'], "nei.pkl"))
            raw_wei = io.load_pickle(osp.join(config['ppr_data_root'], "wei.pkl"))

        topk = config['topk']
        if 'not_include_self' in config['str_set']:
            print("## not_include_self")
            self.nei = torch.LongTensor(raw_nei[:, 1: topk + 1])
            self.wei = torch.FloatTensor(raw_wei[:, 1: topk + 1])
        else:
            self.nei = torch.LongTensor(raw_nei[:, : topk])
            self.wei = torch.FloatTensor(raw_wei[:, : topk])
        
        if 'not_uniform_weight' in config['str_set']:
            print("## not_uniform_weight")
            self.wei = self.wei / (self.wei.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            _w = torch.ones(self.nei.shape)
            _w[self.wei == 0] = 0
            self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)
        # self.wei = self.wei / topk
        
        if self.config['final_layer_mlp']:
            self.mlp = MLP(self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.mlp.parameters(), 'lr': 0.001})
        
        if self.config['learnable_scalar']:
            self.learnable_scalar = LearnableScalar(num_nodes=len(self.base_emb_table), config=self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.learnable_scalar.parameters(), 'lr': 0.001})

        if self.config['degree_mlp']:
            g = io.load_pickle(osp.join(self.config['data_root'], "train_undi_graph.np_csr.pkl"))
            degrees = g['ptr_array'][:, 1]
            self.degree_mlp = DegreeMLP(degrees).to(self.device)
            self.opt_param_list['dense'].append({'params': self.degree_mlp.parameters(), 'lr': 0.001})
        
        self.in_degrees = data['in_degrees']
        
        ## degree discriminator
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.degree_discriminator = DegreeDiscriminator(self.in_degrees).to(self.device)
            self.degree_discriminator_opt = torch.optim.Adam(
                self.degree_discriminator.parameters(), lr=0.001
            )
        
    def _calc_pprgo_out_emb(self, nids):
        top_nids = self.nei[nids]
        top_weights = self.wei[nids]
        
        top_embs = self.base_emb_table[top_nids].to(self.device)
        top_weights = top_weights.to(self.device)
        
        out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        return out_embs.squeeze()
    
    def forward(self, batch_data):
        src, pos, neg = batch_data
        src_emb = self._calc_pprgo_out_emb(src)
        pos_emb = self._calc_pprgo_out_emb(pos)
        neg_emb = self._calc_pprgo_out_emb(neg)
        
        if self.config['final_layer_mlp']:
            if self.config['cat_self']:
                src_emb = self.mlp(src_emb, self_emb=self.base_emb_table[src])
                pos_emb = self.mlp(pos_emb, self_emb=self.base_emb_table[pos])
                neg_emb = self.mlp(neg_emb, self_emb=self.base_emb_table[neg])
            else:
                src_emb = self.mlp(src_emb)
                pos_emb = self.mlp(pos_emb)
                neg_emb = self.mlp(neg_emb)
        
        if self.config['learnable_scalar']:
            src_emb = self.learnable_scalar(src_emb, src)
            pos_emb = self.learnable_scalar(pos_emb, pos)
            neg_emb = self.learnable_scalar(neg_emb, neg)
        
        if self.config['degree_mlp']:
            src_emb = self.degree_mlp(src_emb, src)
            pos_emb = self.degree_mlp(pos_emb, pos)
            neg_emb = self.degree_mlp(neg_emb, neg)
            
        pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, pos_emb, neg_emb)
        loss = bpr_loss(pos_score, neg_score)
        
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            batch_embs = torch.cat((src_emb, pos_emb, neg_emb))
            nids = torch.cat((src, pos, neg))
            degrees = self.in_degrees[nids]
            
            self.degree_discriminator_opt.zero_grad()
            
            pred_logits, true_logits = self.degree_discriminator(
                batch_embs.detach(), degrees)
            discriminator_loss = self.bce_loss(pred_logits, true_logits)
            
            discriminator_loss.backward()
            self.degree_discriminator_opt.step()
            
            # _degree_discriminator = self.degree_discriminator.detach()
            pred_logits, true_logits = self.degree_discriminator(
                batch_embs, degrees)
            _discriminator_loss = self.bce_loss(pred_logits, true_logits)
            
            loss -= self.config['gan_gen_loss_weight'] * _discriminator_loss
        
        return loss
    
    def prepare_for_val(self):
        self.base_emb_table.del_grad_matrix()
        self.out_emb_table = torch.empty(self.base_emb_table.shape, dtype=torch.float32)
        dl = torch.utils.data.DataLoader(dataset=torch.arange(len(self.base_emb_table)), 
                                         batch_size=10240)
        
        for nids in tqdm(dl, desc="get all gnn output embs"):
            self.out_emb_table[nids] = self._calc_pprgo_out_emb(nids).cpu()
        
        if self.config['final_layer_mlp']:
            _mlp = deepcopy(self.mlp).cpu()
            if self.config['scaling']:
                if self.config['cat_self']:
                    self.out_emb_table, self.mlp_output_scalars = \
                        _mlp(self.out_emb_table, self_emb=self.base_emb_table.get_full_matrix().cpu(), 
                             return_scalars=True)
                else:
                    self.out_emb_table, self.mlp_output_scalars = _mlp(self.out_emb_table, return_scalars=True)
            else:
                self.out_emb_table = _mlp(self.out_emb_table, return_scalars=False)
            
            if self.config['use_tao']:
                self.mlp.update_tao()
            
            if self.config['gan_end_to_end']:
                _degree_discriminator = deepcopy(self.degree_discriminator).cpu()
                pred_logits, true_logits = _degree_discriminator(
                    self.out_emb_table, self.in_degrees
                )
                pred_label = pred_logits.argmax(dim=-1)
                true_label = true_logits.argmax(dim=-1)
                precision = (pred_label == true_label).float().mean().item()
                print("## degree_discriminator precision:", precision)
                
                file_gan = osp.join(self.config['results_root'], 'gan_record.txt')
                
                if not osp.exists(file_gan):
                    with open(file_gan, "w") as f:
                        f.write('{:.6g}\n'.format(precision))
                else:    
                    with open(file_gan, "a") as f:
                        f.write('{:.6g}\n'.format(precision))
            
        if self.config['learnable_scalar']:
            _learnable_scalar = deepcopy(self.learnable_scalar).cpu()
            self.out_emb_table = _learnable_scalar(self.out_emb_table)
            
        if self.config['degree_mlp']:
            _degree_mlp = deepcopy(self.degree_mlp).cpu()
            self.out_emb_table = _degree_mlp(self.out_emb_table)
        
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
        torch.save(self.base_emb_table.get_full_matrix().cpu(), osp.join(root, "base_emb_table.pt"))
        if self.config['final_layer_mlp'] and self.config['gan_end_to_end']:
            torch.save(self.degree_discriminator, osp.join(root, "degree_discriminator.pt"))
        if self.config['final_layer_mlp']:
            if not self.config['l2norm']:
                torch.save(self.mlp, osp.join(root, "mlp.pt"))
            if self.config['scaling']:
                torch.save(self.mlp_output_scalars, osp.join(root, "mlp_output_scalars.pt"))
        if self.config['learnable_scalar']:
            torch.save(self.learnable_scalar.w.data.cpu(), osp.join(root, "learnable_scalar.pt"))
        if self.config['degree_mlp']:
            torch.save(self.degree_mlp, osp.join(root, "degree_mlp.pt"))


class FAGCNWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        
        self.build_gnn_and_mlp(GNN=FAGCN, kwargs={"emb_dim":64})   


class SAGEWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        
        arch_list_2layer=[
            {"in_feats": 64, "out_feats": 64, "aggregator_type": 'pool', "activation": torch.tanh},
            {"in_feats": 64, "out_feats": 64, "aggregator_type": 'pool'}
        ]
        arch_list_1layer=[
            {"in_feats": 64, "out_feats": 64, "aggregator_type": 'lstm'},
        ]
        arch = {
            1: arch_list_1layer,
            2: arch_list_2layer,
        }[config['num_gcn_layer']]
        io.save_yaml(osp.join(self.config['results_root'], "sage_arch.yaml"), arch)
        
        self.build_gnn_and_mlp(GNN=SAGE, kwargs={"arch":arch})


class GATWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        
        arch_list_2layer=[
            {"in_feats": 64, "out_feats": 64, "num_heads": 4, "activation": torch.tanh},
            {"in_feats": 64, "out_feats": 64, "num_heads": 4}
        ]
        arch_list_1layer=[
            {"in_feats": 64, "out_feats": 64, "num_heads": 4}
        ]
        arch = {
            1: arch_list_1layer,
            2: arch_list_2layer,
        }[config['num_gcn_layer']]
        io.save_yaml(osp.join(self.config['results_root'], "gat_arch.yaml"), arch)
        
        self.build_gnn_and_mlp(GNN=GAT, kwargs={"arch":arch})


class GINWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        
        self.build_gnn_and_mlp(GNN=GIN, kwargs={"num_gcn_layer": config['num_gcn_layer']})


class SAGNWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        
        self.full_adj = data['full_adj']
        
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')

        self.sagn_mlp_list = torch.nn.ModuleList(
            [TwoLayerMLP() for _ in range(1 + self.config['num_gcn_layer'])]
        ).to(self.device)
        self.att_a = torch.randn(128, 1).to(self.device)
        self.sagn_output_mlp = TwoLayerMLP().to(self.device)
        self.sagn_linear = torch.nn.Linear(64, 64, bias=False).to(self.device)
        
        self.fn_leakyrelu = torch.nn.LeakyReLU()
        self.fn_softmax = torch.nn.Softmax(dim=-1)
        
        self.opt_param_list['dense'].extend([
            {'params': self.sagn_mlp_list.parameters(), 'lr': 0.001},
            {'params': self.att_a, 'lr': 0.001},
            {'params': self.sagn_output_mlp.parameters(), 'lr': 0.001},
            {'params': self.sagn_linear.parameters(), 'lr': 0.001},
        ])
        
        if self.config['final_layer_mlp']:
            self.mlp = MLP(self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.mlp.parameters(), 'lr': 0.001})
        
        elif self.config['each_layer_mlp']:
            self.mlp_list = torch.nn.ModuleList([
                MLP(self.config).to(self.device) for _ in range(1 + self.config['num_gcn_layer'])
            ])
            self.opt_param_list['dense'].append({'params': self.mlp_list.parameters(), 'lr': 0.001})
    
    def _block_update_all(self, blocks, x):
        blocks = [block.to(self.device) for block in blocks]
        for i in range(len(blocks)):
            blocks[i].srcdata['h'] = x
            blocks[i].update_all(self.gcn_msg, self.gcn_reduce)
            x = blocks[i].dstdata['h']
        return x
    
    def _att_2layer(self, X0, X1, X2, infer=False):
        if self.config['each_layer_mlp']:
            if infer:
                X0 = deepcopy(self.mlp_list[0]).cpu()(X0)
                X1 = deepcopy(self.mlp_list[1]).cpu()(X1)
                X2 = deepcopy(self.mlp_list[2]).cpu()(X2)
            else:
                X0 = self.mlp_list[0](X0)
                X1 = self.mlp_list[1](X1)
                X2 = self.mlp_list[1](X2)
        
        if infer:
            a = deepcopy(self.att_a).cpu()
        else:
            a = self.att_a
        
        b0 = torch.cat((X0, X0), dim=-1).matmul(a)
        b1 = torch.cat((X0, X1), dim=-1).matmul(a)
        b2 = torch.cat((X0, X2), dim=-1).matmul(a)
        B = torch.cat((b0, b1, b2), dim=-1)
        B = self.fn_softmax(self.fn_leakyrelu(B))
        
        X_att_out = B[:, 0].unsqueeze(dim=-1) * X0 + \
                    B[:, 1].unsqueeze(dim=-1) * X1 + \
                    B[:, 2].unsqueeze(dim=-1) * X2
        return X_att_out
    
    def _att_1layer(self, X0, X1, infer=False):
        if self.config['each_layer_mlp']:
            if infer:
                X0 = deepcopy(self.mlp_list[0]).cpu()(X0)
                X1 = deepcopy(self.mlp_list[1]).cpu()(X1)
            else:
                X0 = self.mlp_list[0](X0)
                X1 = self.mlp_list[1](X1)
        
        if infer:
            a = deepcopy(self.att_a).cpu()
        else:
            a = self.att_a
        
        b0 = torch.cat((X0, X0), dim=-1).matmul(a)
        b1 = torch.cat((X0, X1), dim=-1).matmul(a)
        B = torch.cat((b0, b1), dim=-1)
        B = self.fn_softmax(self.fn_leakyrelu(B))
        
        X_att_out = B[:, 0].unsqueeze(dim=-1) * X0 + \
                    B[:, 1].unsqueeze(dim=-1) * X1
        return X_att_out
    
    def _compute_2layer(self, batch_data):
        batch_nids, local_idx, B1, B2 = batch_data
        # fetch layer0
        input_nids, output_nids, blocks = B1
        X0 = self.base_emb_table[batch_nids]
        
        # calc layer1
        output_embs = self._block_update_all(blocks, self.base_emb_table[input_nids])
        X1 = output_embs[local_idx]
        
        # calc layer2
        input_nids, output_nids, blocks = B2
        output_embs = self._block_update_all(blocks, self.base_emb_table[input_nids])
        X2 = output_embs[local_idx]
        
        X_att_out = self._att_2layer(X0, X1, X2)
        
        # final output emb
        output_embs = self.sagn_output_mlp(X_att_out + self.sagn_linear(X0))
        
        if self.config['final_layer_mlp']:
            output_embs = self.mlp(output_embs)
        
        return output_embs
    
    def _compute_1layer(self, batch_data):
        batch_nids, local_idx, input_nids, output_nids, blocks = batch_data
        # fetch layer0
        X0 = self.base_emb_table[batch_nids]
        
        # calc layer1
        output_embs = self._block_update_all(blocks, self.base_emb_table[input_nids])
        X1 = output_embs[local_idx]
        
        X_att_out = self._att_1layer(X0, X1)
        
        # final output emb
        output_embs = self.sagn_output_mlp(X_att_out + self.sagn_linear(X0))
        
        if self.config['final_layer_mlp']:
            output_embs = self.mlp(output_embs)
            
        return output_embs
        
    def forward(self, batch_data):
        if self.config['num_gcn_layer'] == 2:
            output_embs = self._compute_2layer(batch_data)
        else:
            output_embs = self._compute_1layer(batch_data)
        
        output_embs = output_embs.view(3, -1, self.base_emb_table.shape[-1])
        
        src_emb = output_embs[0, :, :]
        pos_emb = output_embs[1, :, :]
        neg_emb = output_embs[2, :, :]
        
        pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, pos_emb, neg_emb)
        loss = bpr_loss(pos_score, neg_score)
        
        return loss
    
    def prepare_for_val(self):
        self.base_emb_table.del_grad_matrix()
        
        if self.config['num_gcn_layer'] == 2:
            X0 = self.base_emb_table.get_full_matrix().cpu()
            X1 = torch.sparse.mm(self.full_adj, X0)
            X2 = torch.sparse.mm(self.full_adj, X1)
            X_att_out = self._att_2layer(X0, X1, X2, infer=True)
        else:
            X0 = self.base_emb_table.get_full_matrix().cpu()
            X1 = torch.sparse.mm(self.full_adj, X0)
            X_att_out = self._att_1layer(X0, X1, infer=True)
        
        output_embs = deepcopy(self.sagn_output_mlp).cpu()(
            X_att_out + deepcopy(self.sagn_linear).cpu()(X0))
        
        if self.config['final_layer_mlp']:
            output_embs = deepcopy(self.mlp).cpu()(output_embs)
        
        self.out_emb_table = output_embs.cpu()
        
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
        torch.save(self.base_emb_table.get_full_matrix().cpu(), osp.join(root, "base_emb_table.pt"))
        if self.config['final_layer_mlp']:
            torch.save(self.mlp, osp.join(root, "mlp.pt"))
        if self.config['each_layer_mlp']:
            torch.save(self.mlp_list, osp.join(root, "mlp_list.pt"))


class ClusterGCNWrapper(BaseGNNModel):
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.build_basic(data, config)
        self.full_adj = data['full_adj']
        
        if self.config['final_layer_mlp']:
            self.mlp = MLP(self.config).to(self.device)
            self.opt_param_list['dense'].append({'params': self.mlp.parameters(), 'lr': 0.001})
        
        elif self.config['each_layer_mlp']:
            self.mlp_list = torch.nn.ModuleList([
                MLP(self.config).to(self.device) for _ in range(1 + self.config['num_gcn_layer'])
            ])
            self.opt_param_list['dense'].append({'params': self.mlp_list.parameters(), 'lr': 0.001})
    
    def _compute(self, adj, X, device):
        out = torch.zeros(X.shape).to(device)
        out += self.mlp_list[0].to(device)(X) if self.config['each_layer_mlp'] else X
        layers = self.config['num_gcn_layer']
        H = X
        for i in range(1, layers + 1):
            H = torch.sparse.mm(adj, H)
            if self.config['each_layer_mlp']:
                H = self.mlp_list[i].to(device)(H)
            out += H
        out /= layers + 1
        return out
    
    def forward(self, batch_data):
        subgraph_nids, sub_adj, src, pos, neg = batch_data
        
        X = self.base_emb_table[subgraph_nids]
        
        X = X.to(self.device)
        sub_adj = sub_adj.to(self.device)
        
        X = self._compute(sub_adj, X, self.device)
        
        src_emb = X[src]
        pos_emb = X[pos]
        neg_emb = X[neg]
        
        if self.config['final_layer_mlp']:
            src_emb = self.mlp(src_emb)
            pos_emb = self.mlp(pos_emb)
            neg_emb = self.mlp(neg_emb)
        
        pos_score, neg_score = gnn_module.dot_prodoct_score(src_emb, pos_emb, neg_emb)
        loss = bpr_loss(pos_score, neg_score)
        
        return loss
     
    def prepare_for_val(self):
        self.base_emb_table.del_grad_matrix()
        
        X = self.base_emb_table.get_full_matrix().cpu()
        self.out_emb_table = self._compute(self.full_adj, X, 'cpu')
        
        if self.config['final_layer_mlp']:
            self.out_emb_table, self.mlp_output_scalars = deepcopy(self.mlp).cpu()(self.out_emb_table, return_scalars=True)
    
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
        torch.save(self.base_emb_table.get_full_matrix().cpu(), osp.join(root, "base_emb_table.pt"))
        if self.config['final_layer_mlp']:
            torch.save(self.mlp, osp.join(root, "mlp.pt"))
            torch.save(self.mlp_output_scalars, osp.join(root, "mlp_output_scalars.pt"))
        if self.config['each_layer_mlp']:
            torch.save(self.mlp_list, osp.join(root, "mlp_list.pt"))
