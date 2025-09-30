import torch
import torch.nn as nn

mlp = nn.ModuleList([
    # 第一组：处理3维输入
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 16),

    # 第二组：处理64维输入
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 32),

    # 第三组：处理128维输入
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),

    # 第四组：处理256维输入
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 128),

    # 第五组：处理512维输入（如果需要）
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
])


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

        # 在CCPL的初始化中添加适配器
        self.adapters = nn.ModuleDict()
        self.adapters['3_to_512'] = DimensionAdapter(3, 512)
        self.adapters['64_to_512'] = DimensionAdapter(64, 512)
        self.adapters['128_to_512'] = DimensionAdapter(128, 512)
        self.adapters['256_to_512'] = DimensionAdapter(256, 512)

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n

        # for i in range(3):
        #     feat_d =self.mlp[3*layer+i](feat_d)

        # # 检查特征维度是否与MLP层匹配
        # current_layer = self.mlp[3 * layer]
        # if isinstance(current_layer, nn.Linear) and feat_d.size(-1) != current_layer.in_features:
        #     # 添加适配层
        #     adapter = nn.Linear(feat_d.size(-1), current_layer.in_features).to(feat_d.device)
        #     feat_d = adapter(feat_d)
        #
        # feat_d = feat_d

        for i in range(3):
            # 打印当前层信息
            # current_layer = self.mlp[3 * layer + i]
            # print(f"Layer {3 * layer + i}: {type(current_layer).__name__}")
            #
            # # 如果是线性层，打印权重形状
            # if isinstance(current_layer, nn.Linear):
            #     print(f"  Layer weight shape: {current_layer.weight.shape}")
            #     print(f"  Layer bias shape: {current_layer.bias.shape if current_layer.bias is not None else 'None'}")
            #
            # # 打印输入特征形状
            # print(f"  Input feat_d shape: {feat_d.shape}")

            # current_layer = self.mlp[3 * layer + i]
            # feat_d = current_layer(feat_d)

            feat_d = self.mlp[3 * layer + i](feat_d)


        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
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

    def forward(self, feat_q, feat_k, num_s, tau=0.07):
        loss_ccp = 0.0
        if isinstance(feat_q, list):
            feat_q = feat_q[0]
        if isinstance(feat_k, list):
            feat_k = feat_k[0]

        B, C, H, W = feat_q.shape

        # 根据输入特征维度选择合适的MLP层组
        # 修改选择逻辑，基于通道数C而不是宽度W
        if C == 3:
            i = 0  # 使用第一组MLP层 (64->64->16)
        elif C == 64:
            i = 1  # 使用第二组MLP层 (128->128->32)
        elif C == 128:
            i = 2  # 使用第三组MLP层 (256->256->64)
        elif C == 256:
            i = 3  # 使用第四组MLP层 (512->512->128)
        else:
            # 对于不支持的维度，添加一个适配层
            print(f"Warning: Unsupported feature dimension {C}. Adding adapter layer.")
            # 创建一个适配层，将特征维度转换为512
            adapter = nn.Linear(C, 512).to(feat_q.device)
            feat_q = adapter(feat_q.permute(0, 2, 1)).permute(0, 2, 1)
            feat_k = adapter(feat_k.permute(0, 2, 1)).permute(0, 2, 1)
            i = 3  # 使用第四组MLP层

        f_q, sample_ids = self.NeighborSample(feat_q, i, num_s, [])
        f_k, _ = self.NeighborSample(feat_k, i, num_s, sample_ids)
        loss_ccp = self.PatchNCELoss(f_q, f_k, tau)

        return loss_ccp

    def forward1(self, feat_q, feat_k, num_s, tau=0.07):
        loss_ccp = 0.0
        if isinstance(feat_q, list):
            feat_q = feat_q[0]
        if isinstance(feat_k, list):
            feat_k = feat_k[0]

        B, C, H, W = feat_q.shape

        # 使用适配器将特征维度转换为512
        if C == 3:
            feat_q = self.adapters['3_to_512'](feat_q.permute(0, 2, 1)).permute(0, 2, 1)
            feat_k = self.adapters['3_to_512'](feat_k.permute(0, 2, 1)).permute(0, 2, 1)
        elif C == 64:
            feat_q = self.adapters['64_to_512'](feat_q.permute(0, 2, 1)).permute(0, 2, 1)
            feat_k = self.adapters['64_to_512'](feat_k.permute(0, 2, 1)).permute(0, 2, 1)
        elif C == 128:
            feat_q = self.adapters['128_to_512'](feat_q.permute(0, 2, 1)).permute(0, 2, 1)
            feat_k = self.adapters['128_to_512'](feat_k.permute(0, 2, 1)).permute(0, 2, 1)
        elif C == 256:
            feat_q = self.adapters['256_to_512'](feat_q.permute(0, 2, 1)).permute(0, 2, 1)
            feat_k = self.adapters['256_to_512'](feat_k.permute(0, 2, 1)).permute(0, 2, 1)

        # 现在所有特征都是512维，可以使用第四组MLP层
        i = 3

        f_q, sample_ids = self.NeighborSample(feat_q, i, num_s, [])
        f_k, _ = self.NeighborSample(feat_k, i, num_s, sample_ids)
        loss_ccp = self.PatchNCELoss(f_q, f_k, tau)

        return loss_ccp


class DimensionAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionAdapter, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


