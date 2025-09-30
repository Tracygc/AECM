import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import timm
import time
import torch.nn.functional as F
import sys
from functools import partial
import torch.nn as nn
import math

from torchvision.transforms import transforms as tfs

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class VITCUTModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--adj_size_list', type=list, default=[2, 4, 6, 8, 12],
                            help='different scales of perception field')
        parser.add_argument('--lambda_mlp', type=float, default=1.0, help='weight of lr for discriminator')
        parser.add_argument('--lambda_motion', type=float, default=1.0, help='weight for Temporal Consistency')
        parser.add_argument('--lambda_D_ViT', type=float, default=1.0, help='weight for discriminator')
        parser.add_argument('--lambda_GAN', type=float, default=2.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_global', type=float, default=1.0, help='weight for Global Structural Consistency')
        parser.add_argument('--lambda_spatial', type=float, default=1.0, help='weight for Local Structural Consistency')
        parser.add_argument('--atten_layers', type=str, default='1,3,5,7,9',
                            help='compute Cross-Similarity on which layers')
        parser.add_argument('--local_nums', type=int, default=256)
        parser.add_argument('--which_D_layer', type=int, default=-1)
        parser.add_argument('--side_length', type=int, default=7)

        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]

        if self.isTrain:
            self.model_names = ['G', 'D_ViT']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)

        if self.isTrain:

            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            # self.netPreViT = timm.create_model("vit_base_patch32_384",pretrained=True).to(self.device)
            self.netPreViT = timm.create_model("vit_base_patch16_384", pretrained=True).to(self.device)

            self.norm = F.softmax

            self.resize = tfs.Resize(size=(384, 384))
            # self.resize = tfs.Resize(size=(224, 224))

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for atten_layer in self.atten_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp,
                                                    betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_ViT)

            self.l2norm = Normalize(2)

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
        self.optimizer_D_ViT.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D_ViT.step()

        # update G
        self.set_requires_grad(self.netD_ViT, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        #if self.opt.nce_idt:
        self.idt_B = self.fake[self.real_A.size(0):]

        # print(f"idt_B shape: {self.idt_B.shape}")
        # print(f"idt_B dtype: {self.idt_B.dtype}")

        if self.opt.isTrain:
            real_A = self.real_A
            real_B = self.real_B
            fake_B = self.fake_B
            idt_B = self.idt_B
            self.real_A_resize = self.resize(real_A)
            real_B = self.resize(real_B)
            self.fake_B_resize = self.resize(fake_B)
            self.idt_B_resize = self.resize(idt_B)
            self.mutil_real_A_tokens = self.netPreViT(self.real_A_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_B_tokens = self.netPreViT(real_B, self.atten_layers, get_tokens=True)
            self.mutil_fake_B_tokens = self.netPreViT(self.fake_B_resize, self.atten_layers, get_tokens=True)
            self.mutil_idt_B_tokens = self.netPreViT(self.idt_B_resize, self.atten_layers, get_tokens=True)

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


        fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer].detach()

        real_B_tokens = self.mutil_real_B_tokens[self.opt.which_D_layer]

        fake_B_tokens = self.cat_results(fake_B_tokens, self.opt.adj_size_list)

        real_B_tokens = self.cat_results(real_B_tokens, self.opt.adj_size_list)

        pre_fake_ViT = self.netD_ViT(fake_B_tokens)

        self.loss_D_fake_ViT = self.criterionGAN(pre_fake_ViT, False).mean() * lambda_D_ViT

        pred_real_ViT = self.netD_ViT(real_B_tokens)
        self.loss_D_real_ViT = self.criterionGAN(pred_real_ViT, True).mean() * lambda_D_ViT

        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5

        return self.loss_D_ViT

    def compute_G_loss(self):

        if self.opt.lambda_GAN > 0.0:

            fake_B_tokens = self.mutil_fake_B_tokens[self.opt.which_D_layer]
            fake_B_tokens = self.cat_results(fake_B_tokens, self.opt.adj_size_list)
            pred_fake_ViT = self.netD_ViT(fake_B_tokens)
            self.loss_G_GAN_ViT = self.criterionGAN(pred_fake_ViT, True) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = 0.0

        # if self.opt.lambda_global > 0.0 or self.opt.lambda_spatial > 0.0:
        #     self.loss_global, self.loss_spatial = self.calculate_attention_loss()
        # else:
        #     self.loss_global, self.loss_spatial = 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.mutil_real_A_tokens, self.mutil_fake_B_tokens)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.mutil_real_B_tokens, self.mutil_idt_B_tokens)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        self.loss_G = self.loss_G_GAN_ViT + loss_NCE_both

        return self.loss_G

    def calculate_attention_loss(self):
        n_layers = len(self.atten_layers)
        mutil_real_A_tokens = self.mutil_real_A_tokens
        mutil_fake_B_tokens = self.mutil_fake_B_tokens

        if self.opt.lambda_global > 0.0:
            loss_global = self.calculate_similarity(mutil_real_A_tokens, mutil_fake_B_tokens)


        else:
            loss_global = 0.0

        if self.opt.lambda_spatial > 0.0:
            loss_spatial = 0.0
            local_nums = self.opt.local_nums
            tokens_cnt = 576
            local_id = np.random.permutation(tokens_cnt)
            local_id = local_id[:int(min(local_nums, tokens_cnt))]

            mutil_real_A_local_tokens = self.netPreViT(self.real_A_resize, self.atten_layers, get_tokens=True,
                                                       local_id=local_id, side_length=self.opt.side_length)

            mutil_fake_B_local_tokens = self.netPreViT(self.fake_B_resize, self.atten_layers, get_tokens=True,
                                                       local_id=local_id, side_length=self.opt.side_length)

            loss_spatial = self.calculate_similarity(mutil_real_A_local_tokens, mutil_fake_B_local_tokens)


        else:
            loss_spatial = 0.0

        return loss_global * self.opt.lambda_global, loss_spatial * self.opt.lambda_spatial

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


    def calculate_NCE_loss(self,src, tgt):
        # n_layers = len(self.nce_layers)
        # feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        n_layers = len(self.atten_layers)
        mutil_real_A_tokens = src
        mutil_fake_B_tokens = tgt
        # mutil_real_A_tokens = self.mutil_real_A_tokens
        # mutil_fake_B_tokens = self.mutil_fake_B_tokens
        mutil_real_A_tokens_pool, sample_ids = self.downsample(mutil_real_A_tokens, self.opt.num_patches, None)
        mutil_fake_B_tokens_pool, _ = self.downsample(mutil_fake_B_tokens, self.opt.num_patches, sample_ids)
        # q fakeb  k realA

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
            #feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        # feat_k = self.netG(src, self.nce_layers, encode_only=True)
        # feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        # feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, atten_layers in zip(mutil_fake_B_tokens_pool, mutil_real_A_tokens_pool, self.criterionNCE, self.atten_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        # for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
        #     loss = crit(f_q, f_k) * self.opt.lambda_NCE
        #     total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def downsample(self, tokens, num_patches=256, patch_ids=None):
        return_ids = []
        return_tokens =[]

        for token_id, token in enumerate(tokens):
            B,H,W = token.shape[0], token.shape[1], token.shape[2]
            token_reshape = token.flatten(1,2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[token_id]
                else:
                    patch_id = np.random.permutation(token_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches*num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=token.device)
                token_sample = token_reshape[:, patch_id]
                token_sample = token_sample.view(num_patches, -1)
            else:
                token_sample =token_reshape
                patch_id =[]
            return_ids.append(patch_id)
            token_sample = self.l2norm(token_sample)
            # if num_patches == 0:
            #     token_sample = token_sample.permute(0, 2, 1).reshape([B, token_sample.shape[-1], H, W])
            return_tokens.append(token_sample)
        return return_tokens, return_ids

