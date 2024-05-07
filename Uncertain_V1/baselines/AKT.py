import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.Tcn import TCN
MIN_SEQ_LEN = 5


class AKT(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=128,
        d_fc=256,
        n_heads=8,
        dropout=0.05,
        lambda_cl=0.1,
        shortcut=False,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )
      
        self.lambda_cl=lambda_cl
        self.l_ok=True
        self.h_ok=False
        self.b_ok=False
        self.dropout=nn.Dropout(0.2)
        self.hidden_size=d_model
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.projection_head1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.concatLine=nn.Linear(2*d_model,d_model)
       
    def createLPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        lpFilter = torch.ones((rows, cols))
        lpFilter[d > bandCenter] = 0

        return lpFilter
    
    def createHPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        hpFilter = torch.ones((rows, cols))
        hpFilter[d < bandCenter] = 0

        return hpFilter
    def createBSAilter(self, shape, bandCenter, bandWidth):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        bsFilter = torch.zeros((rows, cols))

        if min(rows, cols) // 2 == bandCenter:
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1
        else:
            bsFilter[d > (bandCenter + bandWidth / 2)] = 1
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1
        return bsFilter
    def forward(self, q_emb, s_emb, lens, n=1):
        # AKT
        hq = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True, n=n)
        hs = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True, n=n)
        h,h_window=self.block3(hq, hq, hs, lens, peek_cur=False, n=n)
        return self.concatLine(torch.cat((h,h_window),2)), hq,hs
    
    def embedding(self,q,s,pid=None):
        lens = (s >= 0).sum(dim=1)
        # set prediction mask
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb
        p_diff=0.0
        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)
            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff
            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff
        
        
        return q_emb, s_emb, lens, p_diff

    def predict(self, q_emb, s_emb,lens, p_diff=None,pid=None, n=1):
    
        h,hq,hs = self(q_emb,s_emb,lens,n)
        
        # hq=self.projection_head(hq)
        # hs=self.projection_head(hs)
        y = self.out(torch.cat([q_emb[:,:, :], h], dim=-1)).squeeze(-1)
        if pid is not None:
            return y, h, (p_diff**2).sum() * 1e-5,hq,hs
        else:
            return y, h, 0.0,hq,hs

    def get_loss(self, q, s, pid=None):
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        logits, _, reg_loss,*_= self.predict(q_emb, s_emb,lens,p_diff,pid)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

    def get_cl_loss(self, q, s, pid=None):
        p_emb_aug_l1,  p_emb_aug_h1, p_emb_aug_b1=[None for i in range(3)]
        bs = s.size(0)
        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid)
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        # augmentation
        q_emb_aug_l1, q_emb_aug_h1,  q_emb_aug_b1=self.augment(q_emb)
        s_emb_aug_l1,  s_emb_aug_h1,  s_emb_aug_b1=self.augment(s_emb)
        if pid is not None:
            p_emb_aug_l1, p_emb_aug_h1, p_emb_aug_b1=self.augment(p_diff)

        # model
        logits, _, reg_loss,hq,hs = self.predict(q_emb, s_emb,lens,p_diff,pid)
        masked_logits = logits[s >= 0]
        cl_losses=.0
        if self.l_ok:
            _, _, _,hq_l,hs_l= self.predict(q_emb_aug_l1, s_emb_aug_l1, lens,p_emb_aug_l1,pid)
            
            cl_loss_ql =self.ncelosss(1,q.device,hq[:, :minlen, :],hq_l[:, :minlen, :])
            cl_loss_sl =self.ncelosss(1,q.device,hs[:, :minlen, :],hs_l[:, :minlen, :])
            cl_losses=cl_loss_ql+cl_loss_sl
        # if self.h_ok:
        #     _, _, _,h_h= self.predict(q_emb_aug_h1, s_emb_aug_h1, lens,p_emb_aug_h1,pid)
        #     cl_loss_h =self.ncelosss(1,q.device,h[:, :minlen, :],h_h[:, :minlen, :])
        #     cl_losses+=cl_loss_h
        # if self.b_ok:
        #     _, _, _,h_b= self.predict(q_emb_aug_b1, s_emb_aug_b1, lens,p_emb_aug_b1,pid)
        #     cl_loss_b =self.ncelosss(1,q.device,h[:, :minlen, :],h_b[:, :minlen, :])
        #     cl_losses+=cl_loss_b
            
        cl_loss=cl_losses/bs
        # prediction loss
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss,reg_loss
    
    
    def ncelosss(self, temperature, device, batch_sample_one, batch_sample_two):
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        b_size = batch_sample_one.shape[0]
        batch_sample_one = batch_sample_one.view(b_size, -1)
        batch_sample_two = batch_sample_two.view(b_size, -1)

        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

    def augment(self, q_emb):
            q_emb_aug_l1, q_emb_aug_l2, q_emb_aug_h1, q_emb_aug_h2, q_emb_aug_b1, q_emb_aug_b2= [None for i in range(6)]

            if self.l_ok:
                q_emb_aug_l1 = self.fft_2(q_emb, self.createLPAilter((q_emb.shape[1], self.hidden_size), q_emb.shape[1]//4))
                q_emb_aug_l1 = self.dropout(q_emb_aug_l1)
            if self.h_ok:
                q_emb_aug_h1 = self.fft_2(q_emb, self.createHPAilter((q_emb.shape[1], self.hidden_size), q_emb.shape[1]//4))
                q_emb_aug_h1 = self.dropout(q_emb_aug_h1)
                
            if self.b_ok:
                BSA=[self.createBSAilter((q_emb.shape[1], self.hidden_size), i, 2)
                    for i in range(min(q_emb.shape[1], self.hidden_size) // 2 + 1)]
                q_emb_aug_b1 = self.fft_2(q_emb,random.choice(BSA) )
                q_emb_aug_b1 = self.dropout(q_emb_aug_b1)
            return q_emb_aug_l1, q_emb_aug_h1, q_emb_aug_b1
        
    def fft_2(self, x, filter):
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift.to(x.device) * filter.to(x.device))))
    
class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)
        
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
     
    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False, n=1):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask_window = torch.tril(torch.ones(seqlen, seqlen), diagonal=-20)
        mask_window=(mask-mask_window).bool()[None, None, :, :].to(self.device())
        mask = mask.bool()[None, None, :, :].to(self.device())
        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0
        
        # apply transformer layer
        if  peek_cur:
            query_= self.masked_attn_head(query, key, values, mask,mask_window,not peek_cur)
            query = query + self.dropout(query_)
            return self.layer_norm(query)
        else:
            query_,query_window= self.masked_attn_head(query, key, values, mask,mask_window,not peek_cur)
            query = query + self.dropout(query_)
            query_window = query + self.dropout(query_window)
            return self.layer_norm(query),self.layer_norm(query_window)


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
            
        self.q_window_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_window_linear = self.q_window_linear
        else:
            self.k_window_linear = nn.Linear(d_model, d_model, bias=bias)
            
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_window=nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask,mask_window,maxout=False):
        bs = q.size(0)
       



        # perform linear operation and split into h heads
        q_ = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k_ = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v_ = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        q_window = self.q_window_linear(q).view(bs, -1, self.h, self.d_k)
        k_window = self.k_window_linear(k).view(bs, -1, self.h, self.d_k)
        

        # transpose to get dimensions bs * h * sl * d_k
        k_ = k_.transpose(1, 2)
        q_ = q_.transpose(1, 2)
        v_ = v_.transpose(1, 2)
        
        k_window = k_window.transpose(1, 2)
        q_window = q_window.transpose(1, 2)
        # v = v.transpose(1, 2)
        
        # calculate attention using function we will define next
        v_1 = attention(
            q_,
            k_,
            v_,
            mask,
            self.gammas,
            maxout
        )
        concat = v_1.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        output_window=None
        if maxout:
            v_window=attention_window(
                q_window,
                k_window,
                v_,
                mask_window,
                self.gammas,
                maxout)
        # concatenate heads and put through final linear layer
            concat_window = v_window.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
            output_window = self.out_proj_window(concat_window)
        if maxout:
            return  output,output_window
        else:
            return output


def attention(q, k, v, mask, gamma=None,maxout=False):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()
    scores = scores.masked_fill(mask == 0, -1e32)

    # include temporal effect
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

        # gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage
    ############################
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale
    # calculate output
    output = torch.matmul(scores, v)
    return output
def attention_window(q, k, v, mask_window,gamma=None,maxout=False):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()
    scores = scores.masked_fill(mask_window == 0, -1e32)

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask_window == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        # gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask_window == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask_window == 0, 0)  # set to hard zero to avoid leakage
    ############################
    # if maxout:
    #     scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
    #     scores *= scale
    # calculate output
    output = torch.matmul(scores, v)
    return output



