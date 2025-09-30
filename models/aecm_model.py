import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import timm
import timm.models.swin_attn as swin_attn
import time
import torch.nn.functional as F
import sys
from functools import partial
import torch.nn as nn
import math
# from .CCPL1 import CCPL, mlp
# from .ccpl2 import CCPL, mlp
from .CCPL3 import CCPL, mlp


from torchvision.transforms import transforms as tfs


class Normalize(nn.Module):
    """从VITCUT添加的Normalize类"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ATENCCPLVITCUTModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT model"""
        # ATENCCPL原有的选项
        parser.add_argument('--adj_size_list', type=list, default=[2, 4, 6, 8, 12],
                            help='different scales of perception field')
        parser.add_argument('--lambda_mlp', type=float, default=1.0, help='weight of lr for discriminator')
        parser.add_argument('--lambda_temporal', type=float, default=1.0, help='weight for Temporal Consistency')
        parser.add_argument('--lambda_D_ViT', type=float, default=1.0, help='weight for discriminator')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_global', type=float, default=1.0, help='weight for Global Structural Consistency')
        parser.add_argument('--lambda_spatial', type=float, default=1.0, help='weight for Local Structural Consistency')
        parser.add_argument('--atten_layers', type=str, default='1,3,5',
                            help='compute Cross-Similarity on which layers')
        parser.add_argument('--local_nums', type=int, default=256)
        parser.add_argument('--which_D_layer', type=int, default=-1)
        parser.add_argument('--side_length', type=int, default=7)

        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute dc loss on which layers')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', type=util.str2bool, nargs='?', const=True,
                            default=False,
                            help='(used for single image translation) If True, include the...)')

        # 从VITCUT添加的选项
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')

        parser.set_defaults(pool_size=0)
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # 损失名称和可视化名称
        self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'D_attn', 'global', 'NCE', 'temporal']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        # 层配置
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        # 添加身份映射损失名称
        if self.opt.nce_idt:
            self.loss_names.append('NCE_Y')

        # 模型名称
        if self.isTrain:
            self.model_names = ['G', 'D_ViT', 'D_attn', 'N_attn', 'S_attn']
        else:
            self.model_names = ['G']

        # 定义网络
        # self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
        #                               opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
        #                               opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
        #                               opt)

        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        # 为可能需要维度转换的层创建投影层
        self.projection_layers = nn.ModuleDict()

        if self.isTrain:
            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            self.netPreViT = timm.create_model("vit_base_patch16_384", pretrained=True).to(self.device)
            self.netD_attn = networks.D_attn().to(self.device)
            self.netN_attn = networks.NCE_attn().to(self.device)
            self.netS_attn = swin_attn.SwinTransformer().to(self.device)

            self.norm = F.softmax
            self.resize = tfs.Resize(size=(384, 384))

            # 从VITCUT添加的Normalize层
            self.l2norm = Normalize(2)

            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            # 为每个注意力层创建NCE损失
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            # 为每个注意力层创建额外的NCE损失（用于ViT token）
            self.criterionNCE_ViT = []
            for atten_layer in self.atten_layers:
                self.criterionNCE_ViT.append(PatchNCELoss(opt).to(self.device))

            self.MSE_loss = nn.MSELoss().to(self.device)
            self.CCPL = CCPL(mlp).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                    betas=(opt.beta1, opt.beta2))
            self.optimizer_D_attn = torch.optim.Adam(self.netD_attn.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                     betas=(opt.beta1, opt.beta2))
            self.optimizer_N_attn = torch.optim.Adam(self.netN_attn.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                     betas=(opt.beta1, opt.beta2))
            self.optimizer_S_attn = torch.optim.Adam(self.netS_attn.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                     betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_ViT)
            self.optimizers.append(self.optimizer_D_attn)
            self.optimizers.append(self.optimizer_N_attn)
            self.optimizers.append(self.optimizer_S_attn)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        pass

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD_ViT, True)
        self.set_requires_grad(self.netD_attn, True)
        self.optimizer_D_ViT.zero_grad()
        self.optimizer_D_attn.zero_grad()

        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()

        self.optimizer_D_ViT.step()
        self.optimizer_D_attn.step()

        # update G
        self.set_requires_grad(self.netD_ViT, False)
        self.set_requires_grad(self.netD_attn, False)
        self.optimizer_G.zero_grad()
        self.optimizer_N_attn.zero_grad()
        self.optimizer_S_attn.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.optimizer_G.step()
        self.optimizer_N_attn.step()
        self.optimizer_S_attn.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    # 保留ATENCCPL原有的方法，同时添加VITCUT的方法
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # 添加VITCUT的身份映射
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        # self.fake_B = self.netG(self.real_A)

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.isTrain:
            real_A = self.real_A
            real_B = self.real_B
            fake_B = self.fake_B
            self.real_A_resize = self.resize(real_A)
            real_B = self.resize(real_B)
            self.fake_B_resize = self.resize(fake_B)

            # 提取token
            self.mutil_real_A_tokens = self.netPreViT(self.real_A_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_B_tokens = self.netPreViT(real_B, self.atten_layers, get_tokens=True)
            self.mutil_fake_B_tokens = self.netPreViT(self.fake_B_resize, self.atten_layers, get_tokens=True)

            # 添加身份映射token提取
            if self.opt.nce_idt:
                idt_B = self.idt_B
                self.idt_B_resize = self.resize(idt_B)
                self.mutil_idt_B_tokens = self.netPreViT(self.idt_B_resize, self.atten_layers, get_tokens=True)

    def tokens_concat(self, origin_tokens, adjacent_size):
        adj_size = adjacent_size
        B, token_num, C = origin_tokens.shape[0], origin_tokens.shape[1], origin_tokens.shape[2]
        S = int(math.sqrt(token_num))
        if S * S != token_num:
            print('Error! Not a square!')
        token_map = origin_tokens.clone().reshape(B,S,S,C)
        cut_patch_list = []
        for i in range(0, S, adj_size):
            for j in range(0, S, adj_size):
                i_left = i
                i_right = i + adj_size + 1 if i + adj_size <= S else S + 1
                j_left = j
                j_right = j + adj_size if j + adj_size <= S else S + 1

                cut_patch = token_map[:, i_left:i_right, j_left: j_right, :]
                cut_patch= cut_patch.reshape(B,-1,C)
                cut_patch = torch.mean(cut_patch, dim=1, keepdim=True)
                cut_patch_list.append(cut_patch)

        result = torch.cat(cut_patch_list,dim=1)
        return result


    def cat_results(self, origin_tokens, adj_size_list):
        res_list = [origin_tokens]
        for ad_s in adj_size_list:
            cat_result = self.tokens_concat(origin_tokens, ad_s)
            res_list.append(cat_result)
        result = torch.cat(res_list, dim=1)

        return result


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""

        lambda_D_ViT = self.opt.lambda_D_ViT
        fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer].detach()
        real_B_tokens = self.mutil_real_B_tokens[self.opt.which_D_layer]

        # 添加了可学习权重的D损失
        fake_B_logit, fake_B_cam_logit = self.netD_attn(fake_B_tokens)
        real_B_logit, real_B_cam_logit = self.netD_attn(real_B_tokens)
        self.loss_D_attn_ad = self.MSE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + \
                              self.MSE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))
        self.loss_D_attn_cam = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

        fake_B_tokens = self.cat_results(fake_B_tokens, self.opt.adj_size_list)
        real_B_tokens = self.cat_results(real_B_tokens, self.opt.adj_size_list)

        # GAN损失中的D损失
        pre_fake_ViT = self.netD_ViT(fake_B_tokens)
        self.loss_D_fake_ViT = self.criterionGAN(pre_fake_ViT, False).mean() * lambda_D_ViT
        pred_real_ViT = self.netD_ViT(real_B_tokens)
        self.loss_D_real_ViT = self.criterionGAN(pred_real_ViT, True).mean() * lambda_D_ViT

        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5
        self.loss_D_attn = (self.loss_D_attn_ad + self.loss_D_attn_cam) * 0.01
        self.loss_D_mix = self.loss_D_ViT + self.loss_D_attn

        return self.loss_D_mix

    def compute_G_loss(self):
        if self.opt.lambda_GAN > 0.0:
            fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer]
            fake_B_tokens = self.cat_results(fake_B_tokens, self.opt.adj_size_list)
            pred_fake_ViT = self.netD_ViT(fake_B_tokens)
            self.loss_G_GAN_ViT = self.criterionGAN(pred_fake_ViT, True) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = 0.0

        if self.opt.lambda_global > 0.0:
            self.loss_global = self.calculate_attention_loss()
        else:
            self.loss_global = 0.0

        if self.opt.lambda_NCE > 0.0:
            # 原有的NCE损失计算（基于生成器特征）
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)

            # 添加VITCUT的NCE损失计算（基于ViT token）
            self.loss_NCE_ViT = self.calculate_NCE_loss_ViT(self.mutil_real_A_tokens, self.mutil_fake_B_tokens)

            # 合并两种NCE损失
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_ViT) * 0.5
        else:
            self.loss_NCE = 0.0

        if self.opt.lambda_temporal > 0.0:
            self.loss_temporal = self.calculate_temporal_loss(self.real_A, self.fake_B)
        else:
            self.loss_temporal = 0.0

        # 添加VITCUT的身份映射损失
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss_ViT(self.mutil_real_B_tokens, self.mutil_idt_B_tokens)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN_ViT + self.loss_global + loss_NCE_both + self.loss_temporal
        return self.loss_G

    # 添加VITCUT的NCE损失计算方法
    def calculate_NCE_loss_ViT(self, src, tgt):
        """VITCUT风格的NCE损失计算，基于ViT token"""
        n_layers = len(self.atten_layers)
        mutil_src_tokens = src
        mutil_tgt_tokens = tgt

        # 使用VITCUT的下采样方法
        mutil_src_tokens_pool, sample_ids = self.downsample(mutil_src_tokens, self.opt.num_patches, None)
        mutil_tgt_tokens_pool, _ = self.downsample(mutil_tgt_tokens, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, atten_layers in zip(mutil_tgt_tokens_pool, mutil_src_tokens_pool, self.criterionNCE_ViT,
                                                self.atten_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def downsample(self, tokens, num_patches=256, patch_ids=None):
        """VITCUT的下采样方法"""
        return_ids = []
        return_tokens = []

        for token_id, token in enumerate(tokens):
            B, H, W = token.shape[0], token.shape[1], token.shape[2]
            token_reshape = token.flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[token_id]
                else:
                    patch_id = np.random.permutation(token_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches * num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=token.device)
                token_sample = token_reshape[:, patch_id]
                token_sample = token_sample.view(num_patches, -1)
            else:
                token_sample = token_reshape
                patch_id = []
            return_ids.append(patch_id)
            token_sample = self.l2norm(token_sample)
            return_tokens.append(token_sample)
        return return_tokens, return_ids

    def calculate_attention_loss(self):
        n_layers = len(self.atten_layers)
        # mutil_real_A_tokens = self.mutil_real_A_tokens
        # mutil_fake_B_tokens = self.mutil_fake_B_tokens
        # 进行滑动自注意机制
        mutil_A_tokens = self.mutil_real_A_tokens
        mutil_B_tokens = self.mutil_fake_B_tokens
        mutil_real_A_tokens = []
        mutil_fake_B_tokens = []
        for src, tgt in zip(mutil_A_tokens, mutil_B_tokens):
            src = self.netS_attn(src)
            tgt = self.netS_attn(tgt)
            mutil_real_A_tokens.append(src)
            mutil_fake_B_tokens.append(tgt)

        if self.opt.lambda_global > 0.0:
            loss_global = self.calculate_similarity(mutil_real_A_tokens, mutil_fake_B_tokens)
        else:
            loss_global = 0.0

        return loss_global * self.opt.lambda_global


    def calculate_similarity(self, mutil_src_tokens, mutil_tgt_tokens):
        loss = 0.0
        n_layers = len(self.atten_layers)
        for src_tokens, tgt_tokens in zip(mutil_src_tokens, mutil_tgt_tokens):
            src_tgt = src_tokens.bmm(tgt_tokens.permute(0,2,1))
            tgt_src = tgt_tokens.bmm(src_tokens.permute(0,2,1))
            cos_dis_global = F.cosine_similarity(src_tgt, tgt_src, dim=-1)
            loss += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
        loss = loss / n_layers
        return loss


    def calculate_NCE_loss(self, src, tgt):

        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        # 加个权重
        feat_q_w = []
        feat_k_w = []
        for q, k, nce_layer in zip(feat_q, feat_k, self.nce_layers):
            q_w, k_w = self.netN_attn(q, k)
            feat_q_w.append(q_w)
            feat_k_w.append(k_w)

        feat_k_pool, sample_ids = self.netF(feat_k_w, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q_w, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def calculate_temporal_loss(self, real, fake):
        # temporal consistency loss
        n_layers = len(self.nce_layers) - 1
        feats_q = self.netG(fake, self.nce_layers, encode_only=True)
        feats_k = self.netG(real, self.nce_layers, encode_only=True)

        loss_temporal = 0.0
        for g_feat, c_feat, nce_layer in zip(feats_q, feats_k, self.nce_layers):
            if nce_layer != 0:
                loss = self.CCPL(g_feat, c_feat, num_s=8)
                loss_temporal += loss
        loss_temporal = self.opt.lambda_temporal * loss_temporal

        return loss_temporal / n_layers



