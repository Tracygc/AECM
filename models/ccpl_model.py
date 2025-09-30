import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .base_model import BaseModel

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)])



class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


# class CCPLModel(nn.Module):
class CCPLModel(BaseModel):

    def __init__(self, mlp):
        super(CCPLModel, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp


    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        ##该函数用于对特征进行采样，返回采样后的向量差。其中feat是输入的特征，layer是当前层的编号，num_s是采样点的数量，sample_ids是已经采样的点的编号。
        ##该函数首先将特征展平，然后随机选择一些点作为锚点，计算这些锚点与其周围点的向量差。接着，将向量差输入到mlp中进行变换，最后对向量进行归一化并返回。
        b, h, w, c = feat.size()
        feat_r = feat.flatten(1, 2)
        if sample_ids == []:
            # 生成采样点的编号
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device)  # 随机生成锚点的编号
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]  # 选择num_s个锚点
            ch_ids = (s_ids // (w - 2) + 1)  # 计算锚点的中心点的行编号
            cw_ids = (s_ids % (w - 2) + 1)  # 计算锚点的中心点的列编号
            c_ids = (ch_ids * w + cw_ids).repeat(8)  # 计算锚点的编号
            delta = [dic[i // num_s] for i in range(8 * num_s)]  # 计算锚点周围点的编号
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta  # 计算周围点的编号
            sample_ids += [c_ids]  # 将锚点的编号加入sample_ids
            sample_ids += [n_ids]  # 将周围点的编号加入sample_ids
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n  # 向量减法

        for i in range(5):
            feat_d =self.mlp[i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids
    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        ## 该函数用于计算PatchNCE损失。其中f_q和f_k是输入的向量，tau是温度参数。
        ## 该函数首先计算正样本的内积，然后计算负样本的内积，并将负样本的对角线设置为负无穷。
        ## 接着，将正负样本的内积拼接起来，除以温度参数后，将结果输入到交叉熵损失函数中计算损失。
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, layer, tau=0.07):
        # feats_q是tgt feats_k是src ,num_s是锚点数 layer是第x层特征，tau是温度参数。
        loss_ccp = 0.0
        if layer == 0:
            i = 2
        if layer == 1:
            i = 3
        if layer == 2:
            i = 3
        f_q, sample_ids = self.NeighborSample(feats_q, i, num_s, []) # i是当前层编号
        f_k, _ = self.NeighborSample(feats_k, i, num_s, sample_ids)
        loss_ccp = self.PatchNCELoss(f_q, f_k, tau)

        return loss_ccp
        # for i in range(start_layer, end_layer):
        #     f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
        #     f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
        #     loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        # return loss_ccp