import numpy as np
import os
import cv2
import argparse
import time
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from init_training import ParaEst
from ref_training import TransformerRefinement, PositionalEncoding

class RefinementDataset(Dataset):
    def __init__(self, device, data_path='.', eval=None, eval_alpha=0):
        if eval == None:
            ny_img = np.load(os.path.join(data_path, 'images_ny_test.npy'))
            gt_img = np.load(os.path.join(data_path, 'images_gt_test.npy'))
            alpha = np.load(os.path.join(data_path, 'alpha_test.npy'))
        else:
            ny_img = np.load(os.path.join(data_path, eval, 'images_ny_test_alpha_%d.npy'%eval_alpha))
            gt_img = np.load(os.path.join(data_path, eval, 'images_gt_test_alpha_%d.npy'%eval_alpha))
        self.ny_img = torch.from_numpy(ny_img).float().to(device)
        self.gt_img = torch.from_numpy(gt_img).float().to(device)
        if eval is None:
            self.alpha = torch.from_numpy(alpha).float().to(device)
        else:
            self.alpha = (torch.ones((gt_img.shape[0]))*eval_alpha).float().to(device)
    def __len__(self):
        return self.gt_img.shape[0]
    def __getitem__(self, idx):
        return self.ny_img[idx, :, :, :], self.gt_img[idx, :, :, :], self.alpha[idx]

def CT_Bound(args, cnn, refiner, helper, datasetloader):
    with torch.no_grad():
        total_ssim, total_psnr, total_mse = 0, 0, 0
        total_running_time = 0
        for j, (ny_img, gt_img, alpha) in enumerate(datasetloader):
            print('%dth image:'%(j+1))
            alpha = alpha.item()
            cv2.imwrite('%stest_visualization/%d_col_gt.png'%(args.data_path, j), gt_img.squeeze().detach().cpu().numpy()/alpha*255)
            cv2.imwrite('%stest_visualization/%d_img_ny.png'%(args.data_path, j), ny_img.squeeze().detach().cpu().numpy()/alpha*255)
            start_time = time.time()

            t_img = ny_img.permute(0,3,1,2)
            img_patches = nn.Unfold(args.R, stride=args.stride)(t_img).view(1, 3, 21, 21, 64, 64)
            vec = img_patches.permute(0,4,5,1,2,3).reshape(64 * 64, 3, 21, 21)
            params_est = cnn(vec.to(torch.float32))
            params = params_est.view(1, 64, 64, 5).permute(0,3,1,2).detach()
            angles = params[:, :3, :, :]
            x0y0   = params[:, 3:, :, :]
            angles = torch.remainder(angles, 2 * np.pi)
            angles = torch.sort(angles, dim=1)[0]
            angles = (angles - np.pi) / np.pi
            x0y0 = x0y0 / 3
            params = torch.cat([angles, x0y0], dim=1).permute(0,2,3,1).flatten(start_dim=1,end_dim=2)
            colors = helper(params, ny_img)
            colors = (colors - 6) / 6
            pm = torch.cat([colors.squeeze(0).flatten(start_dim=0,end_dim=1), angles.squeeze(0), x0y0.squeeze(0)], dim=0).permute(1,2,0).view(1,64*64,14)

            est = refiner(pm)
            if not args.metrics:
                col_est, bndry_est = helper(est, ny_img, colors_only=False)
            else:
                col_est, bndry_est, ssim, psnr, mse = helper(est, ny_img, gt_image=gt_img, alpha=alpha, colors_only=False)
                total_ssim += ssim
                total_psnr += psnr
                total_mse += mse
                print('--- color map: SSIM: %.4f, PSNR (dB): %.4f, MSE: %.4f'%(ssim, psnr, mse))

            running_time = time.time() - start_time

            bndry = bndry_est*255
            if args.eval is not None:
                sio.savemat('%stest_mat/%d.mat'%(args.data_path, j), {'img': bndry_est})
            cv2.imwrite('%stest_visualization/%d_ref_bndry.png'%(args.data_path, j), bndry)
            smoothed_img = col_est[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite('%stest_visualization/%d_ref_col.png'%(args.data_path, j), smoothed_img/alpha*255)
            total_running_time += running_time
            print('--- running time: %.3f s'%running_time)

        print('\nAverage running time: %.3f s'%(total_running_time/len(datasetloader)))
        if args.metrics:
            print('Average metrics for color maps: SSIM: %.3f, PSNR (dB): %.3f, MSE: %.3f'%(total_ssim/len(datasetloader), total_psnr/len(datasetloader), total_mse/len(datasetloader)))

class Helper(nn.Module):
    def __init__(self, R, stride, eta, delta, device):
        super().__init__()
        y, x = torch.meshgrid([torch.linspace(-1.0, 1.0, R), \
                                torch.linspace(-1.0, 1.0, R)])
        self.x = x.view(1, R, R, 1, 1).to(device)
        self.y = y.view(1, R, R, 1, 1).to(device)
        self.R = R
        self.batch_size = 1
        self.eta = eta
        self.delta = delta
        self.stride = stride
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

    def calculate_sim(self, tgt_imgs, est_imgs, alpha):
        tgt_imgs_np = tgt_imgs.detach().cpu().numpy() / alpha
        est_imgs_np = est_imgs.detach().cpu().numpy().transpose(0,2,3,1) / alpha
        ssim = 0
        psnr = 0
        mse = 0
        n = tgt_imgs.shape[0]
        for i in range(n):
            tgt_img = cv2.cvtColor(tgt_imgs_np[i,:,:,:], cv2.COLOR_BGR2GRAY)
            est_img = cv2.cvtColor(est_imgs_np[i,:,:,:], cv2.COLOR_BGR2GRAY)
            # pdb.set_trace()
            ssim += compare_ssim(tgt_img, est_img, data_range=1.0)
            psnr += compare_psnr(tgt_imgs_np[i,:,:,:], est_imgs_np[i,:,:,:], data_range=1.0)
            mse += compare_mse(tgt_imgs_np[i,:,:,:], est_imgs_np[i,:,:,:])
        return ssim/n, psnr/n, mse/n

    def forward(self, ests, ny_image, gt_image=None, alpha=None, colors_only=True):
        ests = ests.permute(0,2,1).view(self.batch_size, 5, self.H_patches, self.W_patches)
        angles = torch.remainder((ests[:, :3, :, :] + 1) * np.pi, 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]
        x0y0 = ests[:, 3:, :, :] * 3
        para = torch.cat([angles, x0y0], dim=1)
        self.img_patches = nn.Unfold(self.R, stride=self.stride)(ny_image.permute(0,3,1,2)).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        dists, colors, patches = self.get_dists_and_patches(para)
        if colors_only:
            return colors
        else:
            self.global_boundaries = self.local2global(self.dists2boundaries(dists))
            global_bndry = self.global_boundaries.squeeze().detach().cpu().numpy()
            self.global_image = self.local2global(patches)
            if gt_image is None:
                return self.global_image, global_bndry
            else:
                ssim, psnr, mse = self.calculate_sim(gt_image, self.global_image, alpha)
                return self.global_image, global_bndry, ssim, psnr, mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda', help='Enable cuda')
    parser.add_argument('--R', type=int, default=21, help='Patch size')
    parser.add_argument('--stride', type=int, default=2, help='Patch size')
    parser.add_argument('--data_path', type=str, default='./dataset/refinement/', help='Path of dataset')
    parser.add_argument('--eta', type=float, default=0.01, help='Width parameter for Heaviside functions')
    parser.add_argument('--delta', type=float, default=0.05, help='Delta parameter')
    parser.add_argument('--metrics', type=bool, default=False, help='Choose whether calculate metrics')
    parser.add_argument('--eval', type=str, default=None, choices=[None,'BSDS500','NYUDv2'], help='Whether to evaluate the model with BSDS500 or NYUDv2')
    parser.add_argument('--eval_alpha', type=int, default=0, choices=[2,4,6,8], help='Photon level for evaluation with BSDS500 or NYUDv2')
    args = parser.parse_args()

    device = torch.device(args.cuda)

    dataset_test = RefinementDataset(device, data_path=args.data_path, eval=args.eval, eval_alpha=args.eval_alpha)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=True)

    cnn = ParaEst().to(device)
    cnn.load_state_dict(torch.load('./dataset/best_ran_pretrained_init.pth'))
    cnn.eval()
    refiner = TransformerRefinement(in_parameter_size=14, out_parameter_size=5, device=device).to(device)
    refiner.load_state_dict(torch.load('./dataset/best_ran_pretrained_ref.pth'))
    refiner.eval()
    helper = Helper(args.R, args.stride, args.eta, args.delta, device)
    CT_Bound(args, cnn, refiner, helper, test_loader)
