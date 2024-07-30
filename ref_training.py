import numpy as np
import math
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RefinementDataset(Dataset):
    def __init__(self, device, data_path='.', train=False):
        if train:
            input_para = np.load(os.path.join(data_path, 'param_src_train.npy'))
            target_para = np.load(os.path.join(data_path, 'param_tgt_train.npy'))
            noisy_img = np.load(os.path.join(data_path, 'images_ny_train.npy'))
            gt_img = np.load(os.path.join(data_path, 'images_gt_train.npy'))
            alpha = np.load(os.path.join(data_path, 'alpha_train.npy'))
        else:
            input_para = np.load(os.path.join(data_path, 'param_src_test.npy'))
            target_para = np.load(os.path.join(data_path, 'param_tgt_test.npy'))
            noisy_img = np.load(os.path.join(data_path, 'images_ny_test.npy'))
            gt_img = np.load(os.path.join(data_path, 'images_gt_test.npy'))
            alpha = np.load(os.path.join(data_path, 'alpha_test.npy'))
        self.input_para = torch.from_numpy(input_para).float().flatten(start_dim=1,end_dim=2).to(device)
        self.target_para = torch.from_numpy(target_para).float().flatten(start_dim=1,end_dim=2).to(device)
        self.noisy_img = torch.from_numpy(noisy_img).float().to(device)
        self.gt_img = torch.from_numpy(gt_img).float().to(device)
        self.alpha = torch.from_numpy(alpha).float().to(device)
    def __len__(self):
        return self.input_para.shape[0]
    def __getitem__(self, idx):
        return self.input_para[idx, :, :], self.target_para[idx, :, :], self.noisy_img[idx, :, :, :], self.gt_img[idx, :, :, :], self.alpha[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, stride, device = None):
        super(PositionalEncoding, self).__init__()
        d_model_half = int(d_model / 2)
        position = torch.linspace(0, (max_len-1)*stride, max_len)
        pe = torch.zeros((max_len, max_len, d_model))
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * (-2 * math.log(10000.0) / d_model)).unsqueeze(0).unsqueeze(0)
        pe[:, :, 0:d_model_half:2] = torch.sin(position.unsqueeze(1).unsqueeze(1) * div_term)
        pe[:, :, 1:d_model_half:2] = torch.cos(position.unsqueeze(1).unsqueeze(1) * div_term)
        pe[:, :, d_model_half:d_model:2] = torch.sin(position.unsqueeze(0).unsqueeze(2) * div_term)
        pe[:, :, d_model_half+1:d_model:2] = torch.cos(position.unsqueeze(0).unsqueeze(2) * div_term)
        self.pe = pe.flatten(start_dim=0, end_dim=1).unsqueeze(0).to(device)
    def forward(self, x):
        x += self.pe[:, :x.size(1), :]
        return x

class TransformerRefinement(nn.Module):
    def __init__(self, max_len=64, stride=2, in_parameter_size=14, out_parameter_size=5, d_model=128, nhead=8, num_encoder_layers=8, \
                    dim_feedforward=256, layer_norm_eps=1e-5, batch_first=True, bias=True, device=None):
        super(TransformerRefinement, self).__init__()
        self.in_src_projection = nn.Linear(in_features=in_parameter_size, out_features=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, stride=stride, device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1,
                                                    activation=F.relu, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=False,
                                                    bias=bias, device=device)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.generator = nn.Linear(in_features=d_model, out_features=out_parameter_size)
    def forward(self, src: torch.Tensor):
        src_emb = self.positional_encoding(self.in_src_projection(src))
        outs = self.encoder(src_emb)
        outs = self.generator(outs)
        return outs

def evaluateDataset(args, model, criteria, datasetloader, data_size, init_train=True):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for src, tgt, ny_img, gt_img, alpha in datasetloader:
            est = model(src)
            if not init_train:
                loss = criteria(est, ny_img, gt_img, alpha)
            else:
                loss = criteria(est, tgt)
            total_loss += loss
        num_batch = data_size // args.batch_size
        avg_total_loss = total_loss / num_batch
        return avg_total_loss

def showCurve(args, points, figname):
    plt.figure(figsize = (8,6))
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')
    epoches_num = np.arange(points.shape[0])
    plt.yscale("log")
    plt.plot(epoches_num, points, linestyle='-', color='b', linewidth=2)
    cf = plt.gcf()
    cf.savefig('%s%s.jpg'%(args.data_path, figname), format='jpg', bbox_inches='tight', dpi=600)

class L2Loss(nn.Module):
    def __init__(self, R, batch_size, stride, eta, delta, lmbda_boundary, lmbda_color, device):
        super(L2Loss, self).__init__()
        y, x = torch.meshgrid([torch.linspace(-1.0, 1.0, R), \
                                torch.linspace(-1.0, 1.0, R)])
        self.x = x.view(1, R, R, 1, 1).to(device)
        self.y = y.view(1, R, R, 1, 1).to(device)
        self.R = R
        self.batch_size = batch_size
        self.eta = eta
        self.delta = delta
        self.stride = stride
        self.lmbda_boundary = lmbda_boundary
        self.lmbda_color = lmbda_color
        self.H = 147
        self.W = 147
        self.H_patches = 64
        self.W_patches = 64
        self.num_patches = torch.nn.Fold(output_size=[self.H, self.W],
                                         kernel_size=R,
                                         stride=stride)(torch.ones(1, R**2,
                                                                    self.H_patches * self.W_patches,
                                                                    device=device)).view(self.H, self.W)
    
    def params2dists(self, params, tau=1e-1):
        x0     = params[:, 3, :, :].unsqueeze(1).unsqueeze(1)
        y0     = params[:, 4, :, :].unsqueeze(1).unsqueeze(1)

        angles = torch.remainder(params[:, :3, :, :], 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]
        angle1 = angles[:, 0, :, :].unsqueeze(1).unsqueeze(1)
        angle2 = angles[:, 1, :, :].unsqueeze(1).unsqueeze(1)
        angle3 = angles[:, 2, :, :].unsqueeze(1).unsqueeze(1)
        angle4 = 0.5 * (angle1 + angle3) + \
                     torch.where(torch.remainder(0.5 * (angle1 - angle3), 2 * np.pi) >= np.pi,
                                 torch.ones_like(angle1) * np.pi, torch.zeros_like(angle1))
        def g(dtheta):
            return (dtheta / np.pi - 1.0) ** 35
        
        sgn42 = torch.where(torch.remainder(angle2 - angle4, 2 * np.pi) < np.pi,
                            torch.ones_like(angle2), -torch.ones_like(angle2))
        tau42 = g(torch.remainder(angle2 - angle4, 2*np.pi)) * tau
        dist42 = sgn42 * torch.min( sgn42 * (-torch.sin(angle4) * (self.x - x0) + torch.cos(angle4) * (self.y - y0)),
                                   -sgn42 * (-torch.sin(angle2) * (self.x - x0) + torch.cos(angle2) * (self.y - y0))) + tau42
        
        sgn13 = torch.where(torch.remainder(angle3 - angle1, 2 * np.pi) < np.pi,
                            torch.ones_like(angle3), -torch.ones_like(angle3))
        tau13 = g(torch.remainder(angle3 - angle1, 2*np.pi)) * tau
        dist13 = sgn13 * torch.min( sgn13 * (-torch.sin(angle1) * (self.x - x0) + torch.cos(angle1) * (self.y - y0)),
                                   -sgn13 * (-torch.sin(angle3) * (self.x - x0) + torch.cos(angle3) * (self.y - y0))) + tau13
        return torch.stack([dist13, dist42], dim=1)

    def dists2indicators(self, dists):
        hdists = 0.5 * (1.0 + (2.0 / np.pi) * torch.atan(dists / self.eta))
        return torch.stack([1.0 - hdists[:, 0, :, :, :, :],
                                  hdists[:, 0, :, :, :, :] * (1.0 - hdists[:, 1, :, :, :, :]),
                                  hdists[:, 0, :, :, :, :] *        hdists[:, 1, :, :, :, :]], dim=1)
    
    def get_dists_and_patches(self, params):
        dists = self.params2dists(params)
        wedges = self.dists2indicators(dists)
        colors = (self.img_patches.unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                    (wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        return dists, colors, patches

    def local2global(self, patches):
        N = patches.shape[0]
        C = patches.shape[1]
        return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.R, stride=self.stride)(
            patches.view(N, C*self.R**2, -1)).view(N, C, self.H, self.W) / \
                self.num_patches.unsqueeze(0).unsqueeze(0)

    def dists2boundaries(self, dists):
        d1 = dists[:, 0:1, :, :, :, :]
        d2 = dists[:, 1:2, :, :, :, :]
        minabsdist = torch.where(d1 < 0.0, -d1, torch.where(d2 < 0.0, torch.min(d1, -d2), torch.min(d1, d2)))
        return 1.0 / (1.0 + (minabsdist / self.delta) ** 2)

    def get_boundary_consistency_term(self, dists):
        curr_global_boundaries_patches = nn.Unfold(self.R, stride=self.stride)(
            self.global_boundaries.detach()).view(self.batch_size, 1, self.R, self.R, self.H_patches, self.W_patches)
        local_boundaries = self.dists2boundaries(dists)
        consistency = ((local_boundaries - curr_global_boundaries_patches) ** 2).mean(-3).mean(-3)
        return consistency
    
    def get_color_consistency_term(self, dists, colors, alpha):
        curr_global_image_patches = nn.Unfold(self.R, stride=self.stride)(
            self.global_image.detach()).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        wedges = self.dists2indicators(dists)
        consistency = (wedges.unsqueeze(1) * (
            colors.unsqueeze(-3).unsqueeze(-3) - curr_global_image_patches.unsqueeze(2)) ** 2).mean(-3).mean(-3).sum(1).sum(1) / alpha[:, None, None]
        return consistency
    
    def get_loss(self, dists, colors, patches, alpha):
        loss_per_patch = ((self.gt_img_patches - patches) ** 2).mean(-3).mean(-3).sum(1) / alpha[:, None, None] + \
                            self.lmbda_boundary * self.get_boundary_consistency_term(dists) + \
                            self.lmbda_color * self.get_color_consistency_term(dists, colors, alpha)
        return loss_per_patch
    
    def forward(self, ests, noisy_image, gt_image, alpha):
        ests = ests.permute(0,2,1).view(self.batch_size, 5, self.H_patches, self.W_patches)
        angles = (ests[:, :3, :, :] + 1) * np.pi
        x0y0 = ests[:, 3:, :, :] * 3
        para = torch.cat([angles, x0y0], dim=1)
        self.img_patches = nn.Unfold(self.R, stride=self.stride)(noisy_image.permute(0,3,1,2)).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        self.gt_img_patches = nn.Unfold(self.R, stride=self.stride)(gt_image.permute(0,3,1,2)).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        dists, colors, patches = self.get_dists_and_patches(para)
        self.global_image = self.local2global(patches)
        self.global_boundaries = self.local2global(self.dists2boundaries(dists))
        loss = self.get_loss(dists, colors, patches, alpha).mean()
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda:0', help='Enable cuda')
    parser.add_argument('--epoch_num', type=int, default=1700, help='Number of epoches')
    parser.add_argument('--learning_rate_1', type=float, default=5e-5, help='Learning rate for initial training')
    parser.add_argument('--learning_rate_2', type=float, default=3.5e-4, help='Initial learning rate for late training')
    parser.add_argument('--loss_change', type=int, default=100, help='Epoch to change the loss function')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--R', type=int, default=21, help='Patch size')
    parser.add_argument('--stride', type=int, default=2, help='Stride')
    parser.add_argument('--lambda_boundary', type=float, default=0.5, help='boundary weight in loss function')
    parser.add_argument('--lambda_color', type=float, default=0.1, help='color weight in loss function')
    parser.add_argument('--data_path', type=str, default='./dataset/refinement/', help='Path of dataset')
    parser.add_argument('--input_size', type=int, default=14, help='Input layer size')
    parser.add_argument('--output_size', type=int, default=5, help='Output layer size')
    parser.add_argument('--eta', type=float, default=0.01, help='Width parameter for Heaviside functions')
    parser.add_argument('--delta', type=float, default=0.05, help='Delta in loss function')
    args = parser.parse_args()

    np.random.seed(1869)
    torch.manual_seed(1869)
    device = torch.device(args.cuda)

    dataset_train = RefinementDataset(device, data_path=args.data_path, train=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset_test = RefinementDataset(device, data_path=args.data_path, train=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

    refiner = TransformerRefinement(in_parameter_size = args.input_size, out_parameter_size = args.output_size, device=device).to(device)
    for p in refiner.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.learning_rate_1)
    criteria_1 = nn.MSELoss()
    criteria_2 = L2Loss(args.R, args.batch_size, args.stride, args.eta, args.delta, args.lambda_boundary, args.lambda_color, device)

    lr_updated = 0
    best_avg_loss = np.inf
    best_epoch = 0
    avg_total_loss = np.zeros((args.epoch_num,), dtype=float)
    for epoch in tqdm(range(args.epoch_num)):
        refiner.train()
        if epoch == args.loss_change:
            scheduler_2 = torch.optim.lr_scheduler.CyclicLR(optimizer, args.learning_rate_2*0.5, args.learning_rate_2, step_size_up=200, step_size_down=200, mode='triangular2', cycle_momentum=False)
        if epoch > args.loss_change:
            scheduler_2.step()
        for step, (src, tgt, ny_img, gt_img, alpha) in enumerate(train_loader):
            est = refiner(src)
            optimizer.zero_grad()
            if epoch >= args.loss_change:
                loss = criteria_2(est, gt_img, gt_img, alpha)
            else:
                loss = criteria_1(est, tgt)
            loss.backward()
            optimizer.step()
        if epoch >= args.loss_change:
            avg_total_loss[epoch] = evaluateDataset(args, refiner, criteria_2, test_loader, len(dataset_test), init_train=False)
        else:
            avg_total_loss[epoch] = evaluateDataset(args, refiner, criteria_1, test_loader, len(dataset_test), init_train=True)
        if epoch >= args.loss_change:
            if avg_total_loss[epoch] < best_avg_loss:
                best_avg_loss = avg_total_loss[epoch]
                torch.save(refiner.state_dict(), './dataset/best_ran_ref.pth')
                best_epoch = epoch
    np.save('%sloss_curve_total.npy'%args.data_path, avg_total_loss)
    showCurve(args, avg_total_loss[:args.loss_change], 'loss_curve_total_tf_1')
    showCurve(args, avg_total_loss[args.loss_change:], 'loss_curve_total_tf_2')
    print('-- Best epoch is {}, with average loss of {}'.format(best_epoch, best_avg_loss))
