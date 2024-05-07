import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import networkx as nx

import matplotlib.pyplot as plt
MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=128,
        d_fc=256,
        n_heads=8,
        n_know=16,
        n_layers=1,
        dropout=0.05,
        lambda_nll=0.001,
        sample_num=5,
        c_model=True,
        uc_model=False,
        shortcut=False,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)
        self.sample_num=sample_num
        if n_pid > 0:
           
            
            self.p_embed = nn.Embedding(n_pid + 1, d_model)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = KnowledgeRetrieverLayer(d_model, n_heads, dropout,self.sample_num)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1)
        )
      
        self.dropout_rate = dropout
        self.lambda_nll = lambda_nll
        
        self.n_layers = n_layers
        self.bce_loss = nn.BCELoss(reduction='none')
        self.c_model=c_model
        self.uc_model=uc_model
        
    
    def forward(self, q_emb, s_emb, lens):
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p,q_scores = self.block3(hq, hq, hs, lens, peek_cur=False)
            #######去除编码器
            # p,q_scores = self.block3(q_emb, q_emb, s_emb, lens, peek_cur=False)
            
            return p, q_scores, q_scores

       

    
    def predict(self, q, s, pid=None, n=1):
        
        # q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        lens = (s >= 0).sum(dim=1)
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb
        p_diff = 0.0
        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_emb = self.p_embed(pid)
            q_emb=q_emb+p_emb
            s_emb=s_emb+p_emb
            
        p, q_scores, k_scores = self(q_emb, s_emb, lens)
       
  ##############################预测阶段################################
        y = self.out(torch.cat([p, q_emb.repeat(self.sample_num,1,1,1)], dim=-1)).squeeze(-1)
        y_kc=self.out(torch.cat([p, self.q_embed(q).repeat(self.sample_num,1,1,1)], dim=-1)).squeeze(-1)

        if pid is not None:
            return y, torch.mean(y,dim=0), q_emb, torch.mean(y_kc,dim=0), q_scores
        else:
            return y, torch.mean(y,dim=0), q_emb, torch.mean(y_kc,dim=0), q_scores

    def get_loss(self, q, s, pid=None):
        logits, logits_mean,_,y_kc, nl_loss = self.predict(q, s, pid)
       

        
        ######均值的信息熵
        entroy_logits_mean=entropy(torch.sigmoid(logits_mean),s)
        ################信息熵的均值
        entroy_mean=torch.mean(entropy(torch.sigmoid(logits),s.unsqueeze(0).repeat(self.sample_num,1,1)),dim=0)
        H=entroy_logits_mean-entroy_mean
        weight_h=F.softmax(H,dim=-1)
        # weight_h=H
        mask_weight=weight_h[s>=0]
        masked_labels = s[s >= 0].float()
       
        masked_logits = logits_mean[s >= 0]
        pred_loss=torch.nn.functional.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="none")
        # weighted_loss = torch.mul(pred_loss, 1)
        if self.c_model:
            weighted_loss = torch.mul(pred_loss, (1-mask_weight))
        elif self.uc_model:
            weighted_loss = torch.mul(pred_loss, (1+mask_weight))
        else:
             weighted_loss = pred_loss

        # # 对加权损失求平均值
        pred_loss = torch.mean(weighted_loss)
        return (pred_loss+ self.lambda_nll*nl_loss),pred_loss,0.1*nl_loss,H
def entropy(p,label):
    """计算给定概率p的二进制信息熵"""
    return -(label)*p * torch.log2(p+1e-12) - (1-label)*(1 - p) * torch.log2(1 - p+1e-12)
    

class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())
        # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self.MLP = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.mse_loss = torch.nn.MSELoss()
        self.attdrop=nn.Dropout(0.2)
    def forward(self, q, k, v, mask, maxout=False):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

         
        
        v_, scores = self.attention(
                q,
                k,
                v,
                mask,
                self.gammas,
                maxout,
            )
        
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output,scores
    
    def attention(self,q, k, v, mask, gamma=None, maxout=False):
    # attention score with scaled dot production
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        bs, head, seqlen, _ = scores.size()
        
        #include temporal effect
        if gamma is not None:
            x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
            x2 = x1.transpose(0, 1).contiguous()

            with torch.no_grad():
                scores_ = scores.masked_fill(mask == 0, -1e32)
                scores_ = F.softmax(scores_, dim=-1)
            
                distcum_scores = torch.cumsum(scores_, dim=-1)
                disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
                position_effect = torch.abs(x1 - x2)[None, None, :, :]
                dist_scores = torch.clamp(
                    (disttotal_scores - distcum_scores) * position_effect, min=0.0
                )
                dist_scores = dist_scores.sqrt().detach()

            gamma = -1.0 * gamma.abs().unsqueeze(0)
            total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

            scores *= total_effect

        # normalize attention score
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
        scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage
        if maxout:
            scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
            scores *= scale
        # calculate output
        scores=self.attdrop(scores)
        output = torch.matmul(scores, v)
        return output, scores
class KnowledgeRetrieverLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout,sample_num, kq_same=True):
        super().__init__()
        self.KnowledgeRetriever = KnowledgeRetriever(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.sample_num=sample_num
    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())
        # apply transformer layer
        query_, scores = self.KnowledgeRetriever(
            query, key, values,self.sample_num, mask, maxout=not peek_cur
        )
        query = query.unsqueeze(0).repeat(self.sample_num,1,1,1) + self.dropout(query_)
        return self.layer_norm(query), scores
    
class KnowledgeRetriever(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self.MLP = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.mse_loss = torch.nn.MSELoss()
        self.attdrop=nn.Dropout(0.2)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )
    def forward(self, q, k, v,n_samples, mask, maxout=False):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        output,nll_loss=self.dis_attention(q,k,v,n_samples,mask,self.gammas)
        return output,nll_loss
    
    def dis_attention(self,q,k,v,n_samples,mask,gamma):
        d_k = k.size(-1)
        qk_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        q=q.unsqueeze(3).repeat(1,1,1,q.shape[2],1)
        k=k.unsqueeze(2).repeat(1,1,q.shape[2],1,1)
        temp=torch.cat((q,k),dim=-1)
        scores_mean =self.MLP(temp).squeeze(-1)
        scores_cov =self.MLP2(temp).squeeze(-1).exp()
        
        import numpy as np
        import matplotlib.pyplot as plt

        # # 创建一个随机的[16, 200]的张量作为示例
        

        # # 画热力图
        # plt.figure(figsize=(10, 5))
        # plt.imshow(scores_mean[1,1,:,:].squeeze(0).squeeze(0), aspect='auto', cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title('Heatmap of the Tensor')
        # plt.savefig('/home/q22301155/codedemo/robust/tsne_visualization.png')
        # plt.show()

        
        distribution = torch.distributions.normal.Normal(scores_mean, torch.sqrt(scores_cov))
       
        log_prob = distribution.log_prob(qk_scores)
        nll_loss = -log_prob.mean()
        
          # 你想要进行的采样次数
        sample_tensors=[]
        scores_sample=[]
        for sample in range(n_samples):
            standard_nor_distribution = torch.randn(scores_mean.shape).to(scores_mean.device)
            scores = scores_mean + torch.mul(standard_nor_distribution, torch.sqrt(scores_cov))
            bs, head, seqlen, _ = scores.size()
            gamma=self.gammas
            if True:
                x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
                x2 = x1.transpose(0, 1).contiguous()
                with torch.no_grad():
                    scores_ = scores.masked_fill(mask == 0, -1e32)
                    scores_ = F.softmax(scores_, dim=-1)
                    distcum_scores = torch.cumsum(scores_, dim=-1)
                    disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
                    position_effect = torch.abs(x1 - x2)[None, None, :, :]
                    dist_scores = torch.clamp(
                        (disttotal_scores - distcum_scores) * position_effect, min=0.0
                    )
                    dist_scores = dist_scores.sqrt().detach()
                gamma = -1.0 * gamma.abs().unsqueeze(0)
                total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
                scores *= total_effect

            scores.masked_fill_(mask == 0, -1e32)
            scores = F.softmax(scores, dim=-1)
            
            
            print(f"第{sample}次采样。。。。。。。。。。。")
            plt.figure(figsize=(10, 8))
            plt.imshow(scores[1,1,0:20,0:20].squeeze(0).squeeze(0), aspect='auto', cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xticks(np.arange(0, 20, step=5),fontsize=14)  # 在 0 到 200 的范围内，每隔 50 设置一个刻度

            # 设置 y 轴刻度
            plt.yticks(np.arange(0, 20, step=5),fontsize=14)  # 在 0 到 16 的范围内，每隔 5 设置一个刻度
            # plt.title('Heatmap of the Tensor')
            plt.savefig(f'/home/q22301155/codedemo/robust/sample{sample}.pdf')
            plt.show()
            
           
            
            scores=self.attdrop(scores)
            v_ = torch.matmul(scores, v)
            concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
            output = self.out_proj(concat)
            sample_tensors.append(output)
        # plt.figure(figsize=(8, 8))
        # scores_1=torch.abs(scores_sample[2] - scores_sample[1])
        # plt.imshow(scores_1[1,1,0:20,0:20].squeeze(0).squeeze(0), aspect='auto', cmap='hot', interpolation='nearest')
        # # plt.colorbar()
        # plt.xticks(np.arange(0, 20, step=5),fontsize=14)  # 在 0 到 200 的范围内，每隔 50 设置一个刻度

        # # 设置 y 轴刻度
        # plt.yticks(np.arange(0, 20, step=5),fontsize=14)  # 在 0 到 16 的范围内，每隔 5 设置一个刻度
        # # plt.title('Heatmap of the Tensor')
        # # plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        # plt.savefig('/home/q22301155/codedemo/robust/discrepancy.pdf')
        # plt.show()    
        outputs=torch.stack(sample_tensors)
        return outputs,nll_loss






    