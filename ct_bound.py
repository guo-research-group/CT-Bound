import numpy as np
import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from init_training import ParaEst
from ref_training import TransformerRefinement, PositionalEncoding

class RefinementDataset(Dataset):
    def __init__(self, device, data_path='.'):
        noisy_img = np.load(os.path.join(data_path, 'images_noisy_test.npy'))
        gt_img = np.load(os.path.join(data_path, 'images_gt_test.npy'))
        alpha = np.load(os.path.join(data_path, 'alpha_test.npy'))
        self.noisy_img = torch.from_numpy(noisy_img).float().to(device)
        self.gt_img = torch.from_numpy(gt_img).float().to(device)
        self.alpha = torch.from_numpy(alpha).float().to(device)
    def __len__(self):
        return self.gt_img.shape[0]
    def __getitem__(self, idx):
        return self.noisy_img[idx, :, :, :], self.gt_img[idx, :, :, :], self.alpha[idx]

def CT_Bound(args, cnn, refiner, assistance, datasetloader):
    bndry_gt_0_all = np.load('%sboundary_gt_test_d_0.npy'%args.data_path)
    bndry_gt_1_all = np.load('%sboundary_gt_test_d_1.npy'%args.data_path)
    bndry_gt_2_all = np.load('%sboundary_gt_test_d_2.npy'%args.data_path)
    with torch.no_grad():
        invalid = 0
        for j, (ny_img, gt_img, alpha) in enumerate(datasetloader):
            alpha = alpha.item()

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
            colors = assistance(params, ny_img, gt_img, alpha, colors_only=True)
            colors = (colors - 6) / 6
            pm = torch.cat([colors.squeeze(0).flatten(start_dim=0,end_dim=1), angles.squeeze(0), x0y0.squeeze(0)], dim=0).permute(1,2,0).view(1,64*64,14)

            est = refiner(pm)
            col_est, bndry_est, ssim, psnr, mse = assistance(est, ny_img, gt_img, alpha, colors_only=False)

            bndry_gt_0 = bndry_gt_0_all[j, :, :]
            bndry_gt_1 = bndry_gt_1_all[j, :, :]
            bndry_gt_2 = bndry_gt_2_all[j, :, :]
            bndry_gt_0_mask = cv2.inRange(bndry_gt_0, 0.1, 1)
            bndry_gt_1_mask = cv2.inRange(bndry_gt_1, 0.1, 1)
            bndry_gt_2_mask = cv2.inRange(bndry_gt_2, 0.1, 1)
            bndry_est_0_mask = cv2.inRange(bndry_est[0], 0.1, 1)
            bndry_est_1_mask = cv2.inRange(bndry_est[1], 0.1, 1)
            bndry_est_2_mask = cv2.inRange(bndry_est[2], 0.1, 1)
            bndry_gt_0_id = np.where(bndry_gt_0_mask>0)
            bndry_gt_0_loc = np.concatenate((bndry_gt_0_id[0][:,None],bndry_gt_0_id[1][:,None]), axis=1)
            bndry_gt_1_id = np.where(bndry_gt_1_mask>0)
            bndry_gt_1_loc = np.concatenate((bndry_gt_1_id[0][:,None],bndry_gt_1_id[1][:,None]), axis=1)
            bndry_gt_2_id = np.where(bndry_gt_2_mask>0)
            bndry_gt_2_loc = np.concatenate((bndry_gt_2_id[0][:,None],bndry_gt_2_id[1][:,None]), axis=1)
            bndry_est_0_id = np.where(bndry_est_0_mask>0)
            bndry_est_0_loc = np.concatenate((bndry_est_0_id[0][:,None],bndry_est_0_id[1][:,None]), axis=1)
            bndry_est_1_id = np.where(bndry_est_1_mask>0)
            bndry_est_1_loc = np.concatenate((bndry_est_1_id[0][:,None],bndry_est_1_id[1][:,None]), axis=1)
            bndry_est_2_id = np.where(bndry_est_2_mask>0)
            bndry_est_2_loc = np.concatenate((bndry_est_2_id[0][:,None],bndry_est_2_id[1][:,None]), axis=1)
            if bndry_est_2_loc.shape[0] == 0:
                invalid += 1
                print('Warning: %dth image does not contain enough boundaries to calculate, so skip it.'%(j+1))
                continue
            distance_0 = np.sqrt(((bndry_gt_0_loc[None,:,:] - bndry_est_0_loc[:,None,:])**2).sum(axis=2))
            min_dist_0 = distance_0.min(axis=1)
            mean_dist_0 = min_dist_0.mean()
            distance_1 = np.sqrt(((bndry_gt_1_loc[None,:,:] - bndry_est_1_loc[:,None,:])**2).sum(axis=2))
            min_dist_1 = distance_1.min(axis=1)
            mean_dist_1 = min_dist_1.mean()
            distance_2 = np.sqrt(((bndry_gt_2_loc[None,:,:] - bndry_est_2_loc[:,None,:])**2).sum(axis=2))
            min_dist_2 = distance_2.min(axis=1)
            mean_dist_2 = min_dist_2.mean()
            print('%dth image:'%(j+1))
            print('--- color map: SSIM: %.4f, PSNR: %.4f, MSE: %.4f'%(ssim, psnr, mse))
            print('--- boundary map: D(0): %.4f, D(1): %.4f, D(2): %.4f'%(mean_dist_0, mean_dist_1, mean_dist_2))

            bndry_0 = (bndry_est[0] - bndry_est[0].min())/(bndry_est[0].max() - bndry_est[0].min())*255
            bndry_1 = (bndry_est[1] - bndry_est[1].min())/(bndry_est[1].max() - bndry_est[1].min())*255
            bndry_2 = (bndry_est[2] - bndry_est[2].min())/(bndry_est[2].max() - bndry_est[2].min())*255
            smoothed_img = col_est[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite('%stest/ct_bound/%d_ref_bndry_d_0.jpg'%(args.data_path, j), bndry_0)
            cv2.imwrite('%stest/ct_bound/%d_ref_bndry_d_1.jpg'%(args.data_path, j), bndry_1)
            cv2.imwrite('%stest/ct_bound/%d_ref_bndry_d_2.jpg'%(args.data_path, j), bndry_2)
            cv2.imwrite('%stest/ct_bound/%d_ref_col.jpg'%(args.data_path, j), smoothed_img/alpha*255)

class Assistance(nn.Module):
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
    
    def calculate_sim(self, tgt_imgs, est_imgs):
        tgt_imgs_np = tgt_imgs.detach().cpu().numpy()
        est_imgs_np = est_imgs.detach().cpu().numpy().transpose(0,2,3,1)
        ssim = 0
        psnr = 0
        mse = 0
        n = tgt_imgs.shape[0]
        for i in range(n):
            tgt_img = cv2.cvtColor(tgt_imgs_np[i,:,:,:], cv2.COLOR_BGR2GRAY)
            est_img = cv2.cvtColor(est_imgs_np[i,:,:,:], cv2.COLOR_BGR2GRAY)
            ssim += compare_ssim(tgt_img, est_img, data_range=17)
            psnr += compare_psnr(tgt_imgs_np[i,:,:,:], est_imgs_np[i,:,:,:], data_range=17)
            mse += compare_mse(tgt_imgs_np[i,:,:,:], est_imgs_np[i,:,:,:])
        return ssim/n, psnr/n, mse/n
    
    def col_diff(self, colors, thres):
        col_diff_01 = torch.sqrt(((colors[:, :, 1, :, :] - colors[:, :, 0, :, :])**2).sum(1))
        col_diff_12 = torch.sqrt(((colors[:, :, 2, :, :] - colors[:, :, 1, :, :])**2).sum(1))
        col_diff_20 = torch.sqrt(((colors[:, :, 0, :, :] - colors[:, :, 2, :, :])**2).sum(1))
        edge_01 = torch.where(col_diff_01>thres, 1, 0)
        edge_12 = torch.where(col_diff_12>thres, 2, 0)
        edge_20 = torch.where(col_diff_20>thres, 4, 0)
        indicator = edge_01 + edge_12 + edge_20
        return indicator

    def modify_para(self, param_org, indicator):
        param = param_org.clone()
        case_0_id = torch.where(indicator==0)
        case_1_id = torch.where(indicator==1)
        case_2_id = torch.where(indicator==2)
        case_3_id = torch.where(indicator==3)
        case_4_id = torch.where(indicator==4)
        case_5_id = torch.where(indicator==5)
        case_6_id = torch.where(indicator==6)

        param[case_0_id[0], :3, case_0_id[1], case_0_id[2]] = 0
        param[case_0_id[0], 3:, case_0_id[1], case_0_id[2]] = -3

        param[case_1_id[0], 1, case_1_id[1], case_1_id[2]] = param[case_1_id[0], 2, case_1_id[1], case_1_id[2]]
        param[case_1_id[0], 0, case_1_id[1], case_1_id[2]] = param[case_1_id[0], 2, case_1_id[1], case_1_id[2]]

        param[case_2_id[0], 2, case_2_id[1], case_2_id[2]] = param[case_2_id[0], 1, case_2_id[1], case_2_id[2]]
        param[case_2_id[0], 0, case_2_id[1], case_2_id[2]] = param[case_2_id[0], 1, case_2_id[1], case_2_id[2]]

        param[case_3_id[0], 0, case_3_id[1], case_3_id[2]] = param[case_3_id[0], 2, case_3_id[1], case_3_id[2]]

        param[case_4_id[0], 2, case_4_id[1], case_4_id[2]] = param[case_4_id[0], 0, case_4_id[1], case_4_id[2]]
        param[case_4_id[0], 1, case_4_id[1], case_4_id[2]] = param[case_4_id[0], 0, case_4_id[1], case_4_id[2]]

        param[case_5_id[0], 1, case_5_id[1], case_5_id[2]] = param[case_5_id[0], 0, case_5_id[1], case_5_id[2]]

        param[case_6_id[0], 2, case_6_id[1], case_6_id[2]] = param[case_6_id[0], 1, case_6_id[1], case_6_id[2]]

        return param

    def forward(self, ests, noisy_image, gt_image, alpha, colors_only=True):
        ests = ests.permute(0,2,1).view(self.batch_size, 5, self.H_patches, self.W_patches)
        angles = torch.remainder((ests[:, :3, :, :] + 1) * np.pi, 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]
        x0y0 = ests[:, 3:, :, :] * 3
        para = torch.cat([angles, x0y0], dim=1)
        para_bdry = torch.cat([angles, x0y0], dim=1)
        self.img_patches = nn.Unfold(self.R, stride=self.stride)(noisy_image.permute(0,3,1,2)).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        self.gt_img_patches = nn.Unfold(self.R, stride=self.stride)(gt_image.permute(0,3,1,2)).view(self.batch_size, 3, self.R, self.R, self.H_patches, self.W_patches)
        dists, colors, patches = self.get_dists_and_patches(para)
        if colors_only:
            return colors
        else:
            self.global_boundaries = self.local2global(self.dists2boundaries(dists))
            global_bndry = [self.global_boundaries.squeeze().detach().cpu().numpy()]
            for thres in [0.1, 0.2]:
                indicator = self.col_diff(colors, thres*alpha)
                para_new = self.modify_para(para_bdry, indicator)
                dists_filter, _, _ = self.get_dists_and_patches(para_new)
                global_bndry.append(self.local2global(self.dists2boundaries(dists_filter)).squeeze().detach().cpu().numpy().copy())
            self.global_image = self.local2global(patches)
            ssim, psnr, mse = self.calculate_sim(gt_image, self.global_image)
            return self.global_image, global_bndry, ssim, psnr, mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda:0', help='Enable cuda')
    parser.add_argument('--R', type=int, default=21, help='Patch size')
    parser.add_argument('--stride', type=int, default=2, help='Patch size')
    parser.add_argument('--data_path', type=str, default='./dataset/refinement/', help='Path of dataset')
    parser.add_argument('--eta', type=float, default=0.01, help='Width parameter for Heaviside functions')
    parser.add_argument('--delta', type=float, default=0.05, help='delta parameter')
    args = parser.parse_args()

    np.random.seed(1896)
    torch.manual_seed(1896)
    device = torch.device(args.cuda)

    dataset_test = RefinementDataset(device, data_path=args.data_path)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=True)

    cnn = ParaEst().to(device)
    cnn.load_state_dict(torch.load('./dataset/initialization/best_ran_pretrained_init.pth'))
    cnn.eval()
    refiner = TransformerRefinement(in_parameter_size=14, out_parameter_size=5, device=device).to(device)
    refiner.load_state_dict(torch.load('./dataset/refinement/best_ran_pretrained_ref.pth'))
    refiner.eval()
    assistance = Assistance(args.R, args.stride, args.eta, args.delta, device)
    CT_Bound(args, cnn, refiner, assistance, test_loader)