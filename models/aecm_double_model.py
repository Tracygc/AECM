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

import random
from matplotlib import pyplot as plt
import os
import datetime

from torchvision.transforms import transforms as tfs


class AECMDOUBLEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
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

        parser.add_argument('--temporal_loss_mode', type=int, default='1', choices=['0', '1', '2'],
                            help='which kind of temporal consistency')
        parser.add_argument('--k_sizes', type=list, default=[1, 3, 5, 7],
                            help='kernel sizes for multi-scale relation-based loss')
        parser.add_argument('--weight_t', type=float, default=1.0, help='weight for temporal consistency loss')
        parser.add_argument('--k_weights', type=list, default=[0.25, 0.25, 0.25, 0.25],
                            help='weights used to blend different scales')

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'D_attn', 'global', 'NCE', 'temporal']
        self.visual_names = ['real_A0', 'real_A1', 'fake_B0', 'fake_B1', 'real_B0', 'real_B1']
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.isTrain:
            self.model_names = ['G', 'D_ViT', 'D_attn', 'N_attn', 'S_attn']
        else:  # during test time, only load Gmotion
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)

        if self.isTrain:

            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            self.netPreViT = timm.create_model("vit_base_patch16_384", pretrained=True).to(self.device)
            #  带可学习权重的鉴别器网络
            self.netD_attn = networks.D_attn().to(self.device)
            #  带可学习权重的特征加权网络
            self.netN_attn = networks.NCE_attn().to(self.device)
            #  滑动窗口自注意
            self.netS_attn = swin_attn.SwinTransformer().to(self.device)

            self.norm = F.softmax
            self.resize = tfs.Resize(size=(384, 384))

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.MSE_loss = nn.MSELoss().to(self.device)

            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                    betas=(opt.beta1, opt.beta2))
            # add
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
        self.real_A0 = input['A0' if AtoB else 'B0'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        self.real_B0 = input['B0' if AtoB else 'A0'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B0 = self.netG(self.real_A0)
        self.fake_B1 = self.netG(self.real_A1)

        if self.opt.isTrain:
            real_A0 = self.real_A0
            real_A1 = self.real_A1
            real_B0 = self.real_B0
            real_B1 = self.real_B1
            fake_B0 = self.fake_B0
            fake_B1 = self.fake_B1
            self.real_A0_resize = self.resize(real_A0)
            self.real_A1_resize = self.resize(real_A1)
            real_B0 = self.resize(real_B0)
            real_B1 = self.resize(real_B1)
            self.fake_B0_resize = self.resize(fake_B0)
            self.fake_B1_resize = self.resize(fake_B1)
            self.mutil_real_A0_tokens = self.netPreViT(self.real_A0_resize, self.atten_layers, get_tokens=True)  #3层的1，576，768
            self.mutil_real_A1_tokens = self.netPreViT(self.real_A1_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_B0_tokens = self.netPreViT(real_B0, self.atten_layers, get_tokens=True)
            self.mutil_real_B1_tokens = self.netPreViT(real_B1, self.atten_layers, get_tokens=True)
            self.mutil_fake_B0_tokens = self.netPreViT(self.fake_B0_resize, self.atten_layers, get_tokens=True)
            self.mutil_fake_B1_tokens = self.netPreViT(self.fake_B1_resize, self.atten_layers, get_tokens=True)


    def tokens_concat(self, origin_tokens, adjacent_size):
        adj_size = adjacent_size
        B, token_num, C = origin_tokens.shape[0], origin_tokens.shape[1], origin_tokens.shape[2]
        S = int(math.sqrt(token_num))
        if S * S != token_num:
            print('Error! Not a square!')
        token_map = origin_tokens.clone().reshape(B, S, S, C)
        cut_patch_list = []
        for i in range(0, S, adj_size):
            for j in range(0, S, adj_size):
                i_left = i
                i_right = i + adj_size + 1 if i + adj_size <= S else S + 1
                j_left = j
                j_right = j + adj_size if j + adj_size <= S else S + 1

                cut_patch = token_map[:, i_left:i_right, j_left: j_right, :]
                cut_patch = cut_patch.reshape(B, -1, C)
                cut_patch = torch.mean(cut_patch, dim=1, keepdim=True)
                cut_patch_list.append(cut_patch)

        result = torch.cat(cut_patch_list, dim=1)
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
        fake_B0_tokens = self.mutil_fake_B0_tokens[self.opt.which_D_layer].detach()
        fake_B1_tokens = self.mutil_fake_B1_tokens[self.opt.which_D_layer].detach()

        real_B0_tokens = self.mutil_real_B0_tokens[self.opt.which_D_layer]
        real_B1_tokens = self.mutil_real_B1_tokens[self.opt.which_D_layer]

        # 添加了可学习权重的D损失
        fake_B0_logit, fake_B0_cam_logit = self.netD_attn(fake_B0_tokens)
        real_B0_logit, real_B0_cam_logit = self.netD_attn(real_B0_tokens)
        fake_B1_logit, fake_B1_cam_logit = self.netD_attn(fake_B1_tokens)
        real_B1_logit, real_B1_cam_logit = self.netD_attn(real_B1_tokens)

        self.loss_D0_attn_ad = self.MSE_loss(real_B0_logit, torch.ones_like(real_B0_logit).to(self.device)) + \
                               self.MSE_loss(fake_B0_logit, torch.zeros_like(fake_B0_logit).to(self.device))
        self.loss_D0_attn_cam = self.MSE_loss(real_B0_cam_logit, torch.ones_like(real_B0_cam_logit).to(self.device)) + \
                                self.MSE_loss(fake_B0_cam_logit, torch.zeros_like(fake_B0_cam_logit).to(self.device))

        self.loss_D1_attn_ad = self.MSE_loss(real_B1_logit, torch.ones_like(real_B1_logit).to(self.device)) + \
                               self.MSE_loss(fake_B1_logit, torch.zeros_like(fake_B1_logit).to(self.device))
        self.loss_D1_attn_cam = self.MSE_loss(real_B1_cam_logit, torch.ones_like(real_B1_cam_logit).to(self.device)) + \
                                self.MSE_loss(fake_B1_cam_logit, torch.zeros_like(fake_B1_cam_logit).to(self.device))

        self.loss_D_attn_ad = (self.loss_D0_attn_ad + self.loss_D1_attn_ad) * 0.5
        self.loss_D_attn_cam = (self.loss_D0_attn_cam + self.loss_D1_attn_cam) * 0.5

        fake_B0_tokens = self.cat_results(fake_B0_tokens, self.opt.adj_size_list)
        fake_B1_tokens = self.cat_results(fake_B1_tokens, self.opt.adj_size_list)

        real_B0_tokens = self.cat_results(real_B0_tokens, self.opt.adj_size_list)
        real_B1_tokens = self.cat_results(real_B1_tokens, self.opt.adj_size_list)

        # GAN损失中的D损失
        pre_fake0_ViT = self.netD_ViT(fake_B0_tokens)
        pre_fake1_ViT = self.netD_ViT(fake_B1_tokens)
        self.loss_D_fake_ViT = (self.criterionGAN(pre_fake0_ViT, False).mean() + self.criterionGAN(pre_fake1_ViT,
                                                                                                   False).mean()) * 0.5 * lambda_D_ViT

        pred_real0_ViT = self.netD_ViT(real_B0_tokens)
        pred_real1_ViT = self.netD_ViT(real_B1_tokens)
        self.loss_D_real_ViT = (self.criterionGAN(pred_real0_ViT, True).mean() + self.criterionGAN(pred_real1_ViT,
                                                                                                   True).mean()) * 0.5 * lambda_D_ViT

        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5
        self.loss_D_attn = (self.loss_D_attn_ad + self.loss_D_attn_cam) * 0.01
        self.loss_D_mix = self.loss_D_ViT + self.loss_D_attn

        return self.loss_D_mix

    def compute_G_loss(self):

        if self.opt.lambda_GAN > 0.0:
            fake_B0_tokens = self.mutil_fake_B0_tokens[self.opt.which_D_layer]
            fake_B1_tokens = self.mutil_fake_B1_tokens[self.opt.which_D_layer]
            fake_B0_tokens = self.cat_results(fake_B0_tokens, self.opt.adj_size_list)
            fake_B1_tokens = self.cat_results(fake_B1_tokens, self.opt.adj_size_list)
            pred_fake0_ViT = self.netD_ViT(fake_B0_tokens)
            pred_fake1_ViT = self.netD_ViT(fake_B1_tokens)
            self.loss_G_GAN_ViT = (self.criterionGAN(pred_fake0_ViT, True) + self.criterionGAN(pred_fake1_ViT,
                                                                                               True)) * 0.5 * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = 0.0

        if self.opt.lambda_global > 0.0:
            self.loss_global = self.calculate_attention_loss()
        else:
            self.loss_global = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = (self.calculate_NCE_loss(self.real_A0, self.fake_B0) +
                             self.calculate_NCE_loss(self.real_A1, self.fake_B1)) * 0.5  # * self.opt.lambda_NCE
        else:
            self.loss_NCE = 0.0

        if self.opt.lambda_temporal > 0.0:
            self.loss_temporal = self.calculate_temporal_loss()
        else:
            self.loss_temporal = 0.0

        self.loss_G = self.loss_G_GAN_ViT + self.loss_global + self.loss_NCE + self.loss_temporal
        return self.loss_G

    def calculate_attention_loss(self):
        n_layers = len(self.atten_layers)
        # 进行滑动自注意机制
        mutil_A0_tokens = self.mutil_real_A0_tokens
        mutil_B0_tokens = self.mutil_fake_B0_tokens
        mutil_A1_tokens = self.mutil_real_A1_tokens
        mutil_B1_tokens = self.mutil_fake_B1_tokens
        mutil_real_A0_tokens = []
        mutil_fake_B0_tokens = []
        mutil_real_A1_tokens = []
        mutil_fake_B1_tokens = []
        for src, tgt, srcc, tgtt, in zip(mutil_A0_tokens, mutil_B0_tokens, mutil_A1_tokens, mutil_B1_tokens):
            src = self.netS_attn(src)  # print(src.shape) 1,576,768
            tgt = self.netS_attn(tgt)
            srcc = self.netS_attn(srcc)
            tgtt = self.netS_attn(tgtt)
            mutil_real_A0_tokens.append(src)
            mutil_fake_B0_tokens.append(tgt)
            mutil_real_A1_tokens.append(srcc)
            mutil_fake_B1_tokens.append(tgtt)

        if self.opt.lambda_global > 0.0:
            loss_global = (self.calculate_similarity(mutil_real_A0_tokens, mutil_fake_B0_tokens) +
                           self.calculate_similarity(mutil_real_A1_tokens, mutil_fake_B1_tokens)) * 0.5
        else:
            loss_global = 0.0

        return loss_global * self.opt.lambda_global

    def calculate_similarity(self, mutil_src_tokens, mutil_tgt_tokens):
        loss = 0.0
        n_layers = len(self.atten_layers)
        for src_tokens, tgt_tokens in zip(mutil_src_tokens, mutil_tgt_tokens):
            src_tgt = src_tokens.bmm(tgt_tokens.permute(0, 2, 1))
            tgt_src = tgt_tokens.bmm(src_tokens.permute(0, 2, 1))
            cos_dis_global = F.cosine_similarity(src_tgt, tgt_src, dim=-1)
            loss += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
        loss = loss / n_layers
        return loss

    def random_num(self, size, end):
        range_ls = [i for i in range(end)]
        num_ls = []
        for i in range(size):
            num = random.choice(range_ls)
            range_ls.remove(num)
            num_ls.append(num)
        return num_ls

    def calculate_NCE_loss(self, src, tgt):

        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        # weight add
        feat_q_w = []
        feat_k_w = []
        for q, k, nce_layer in zip(feat_q, feat_k, self.nce_layers):
            q_w, k_w = self.netN_attn(q, k)
            # print(q_w.shape)
            # torch.Size([1, 128, 256, 256]),torch.Size([1, 256, 128, 128]),torch.Size([1, 256, 64, 64])
            # torch.Size([1, 256, 64, 64]),torch.Size([1, 3, 262, 262])

            if q_w.shape[1] > 25:
                # 定义保存图像的基础路径
                base_path = 'G:/pytorch/ATEN_AVIID1-unpaired-8/mapplt'
                # 确保路径存在，如果不存在则创建它
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                # 使用当前时间生成唯一的文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"feature_{timestamp}.jpg"
                # 构建完整的文件路径
                full_path = os.path.join(base_path, filename)
                # 取消Tensor的梯度并转成三维tensor，否则无法绘图
                q_w_t = q_w.data.cpu().numpy()
                q_w_t = q_w_t.squeeze(0)
                # 随机选取25个通道的特征图
                channel_num = self.random_num(25, q_w_t.shape[0])
                plt.figure(figsize=(10, 10))
                for index, channel in enumerate(channel_num):
                    ax = plt.subplot(5, 5, index + 1, )
                    plt.imshow(q_w_t[channel, :, :])
                plt.savefig(full_path, dpi=300)

            feat_q_w.append(q_w)
            feat_k_w.append(k_w)

        feat_k_pool, sample_ids = self.netF(feat_k_w, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q_w, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_temporal_loss(self):

        if self.opt.lambda_temporal > 0.0:
            self.loss_temporal = 0.0
            for real_A0_tokens, real_A1_tokens, fake_B0_tokens, fake_B1_tokens in zip(self.mutil_real_A0_tokens,
                                                                                      self.mutil_real_A1_tokens,
                                                                                      self.mutil_fake_B0_tokens,
                                                                                      self.mutil_fake_B1_tokens):
                A0_B1 = real_A0_tokens.bmm(fake_B1_tokens.permute(0, 2, 1))
                B0_A1 = fake_B0_tokens.bmm(real_A1_tokens.permute(0, 2, 1))
                cos_dis_global = F.cosine_similarity(A0_B1, B0_A1, dim=-1)
                self.loss_temporal += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
            self.loss_temporal = self.loss_temporal * self.opt.lambda_temporal
        else:
            self.loss_temporal = 0.0

        return self.loss_temporal


