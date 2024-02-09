import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

class ParaDataset(Dataset):
    def __init__(self, device, data_path='.', train=False):
        if train:
            X = np.load(os.path.join(data_path, 'patches_noisy_train.npy'))
            y = np.load(os.path.join(data_path, 'patches_gt_train.npy'))
            alpha = np.load(os.path.join(data_path, 'alpha_train.npy'))
        else:
            X = np.load(os.path.join(data_path, 'patches_noisy_test.npy'))
            y = np.load(os.path.join(data_path, 'patches_gt_test.npy'))
            alpha = np.load(os.path.join(data_path, 'alpha_test.npy'))
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.alpha = torch.from_numpy(alpha).float().to(device)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx, :, :, :], self.y[idx, :, :, :], self.alpha[idx]

class ParaEst(nn.Module):
    def __init__(self):
        super(ParaEst, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96, 
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, 
                      out_channels=256, 
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, 
                      out_channels=384, 
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, 
                      out_channels=384, 
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, 
                      out_channels=256, 
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2*2*256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=5)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class L2Loss(nn.Module):
    def __init__(self, R, eta, device):
        super(L2Loss, self).__init__()
        y, x = torch.meshgrid([torch.linspace(-1.0, 1.0, R), \
                                torch.linspace(-1.0, 1.0, R)], indexing='ij')
        self.x = x.view(1, R, R).to(device)
        self.y = y.view(1, R, R).to(device)
        self.eta = eta

    def params2dists(self, params, tau=1e-1):
        x0     = params[:, 3].unsqueeze(1).unsqueeze(1) # shape [N, 1, 1]
        y0     = params[:, 4].unsqueeze(1).unsqueeze(1) # shape [N, 1, 1]

        angles = torch.remainder(params[:, :3], 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]

        angle1 = angles[:, 0].unsqueeze(1).unsqueeze(1) # shape [N, 1, 1]
        angle2 = angles[:, 1].unsqueeze(1).unsqueeze(1) # shape [N, 1, 1]
        angle3 = angles[:, 2].unsqueeze(1).unsqueeze(1) # shape [N, 1, 1]
        
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
        return torch.stack([1.0 - hdists[:, 0, :, :],
                                  hdists[:, 0, :, :] * (1.0 - hdists[:, 1, :, :]),
                                  hdists[:, 0, :, :] *        hdists[:, 1, :, :]], dim=1)
    
    def get_dists_and_patches(self, params, img_ny):
        dists = self.params2dists(params)
        wedges = self.dists2indicators(dists)
        colors = (img_ny.permute(0,3,1,2).unsqueeze(2) * wedges.unsqueeze(1)).sum(-1).sum(-1) / \
                    (wedges.sum(-1).sum(-1).unsqueeze(1) + 1e-10)
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        return patches
    
    def forward(self, params, img_ny, img_gt, alpha):
        patches = self.get_dists_and_patches(params, img_ny).permute(0,2,3,1)
        loss = ((img_gt - patches) ** 2 / alpha[:, None, None, None]).sum(-1).mean(-1).mean(-1).mean(0)
        return loss

def evaluateDataset(args, criteria, model, datasetloader, data_size):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for img_ny, img_gt, alpha in datasetloader:
            est = model(img_ny.permute(0,3,1,2))
            loss = criteria(est, img_ny, img_gt, alpha)
            total_loss += loss
        num_batch = data_size // args.batch_size
        avg_total_loss = total_loss / num_batch
        return avg_total_loss

def showCurve(args, points, figname):
    plt.figure(figsize = (8,6))
    plt.xlabel('Epoches')
    plt.ylabel('Average loss')
    epoches_num = np.arange(points.shape[0])
    plt.yscale("log")
    plt.plot(epoches_num, points, linestyle='-', color='b', linewidth=2)
    cf = plt.gcf()
    cf.savefig('%s%s.jpg'%(args.data_path, figname), format='jpg', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda:0', help='Enable cuda')
    parser.add_argument('--epoch_num', type=int, default=900, help='Number of epoches')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Initial learning rate for late training')
    parser.add_argument('--lr_update', type=int, default=80, help='Number of epochs to update the learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size')
    parser.add_argument('--R', type=int, default=21, help='Patch size')
    parser.add_argument('--eta', type=int, default=0.01, help='eta in loss function')
    parser.add_argument('--data_path', type=str, default='./dataset/initialization/', help='Path of dataset')
    args = parser.parse_args()
    
    np.random.seed(1896)
    torch.manual_seed(1896)
    device = torch.device(args.cuda)
    dataset_train = ParaDataset(device, data_path=args.data_path, train=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset_test = ParaDataset(device, data_path=args.data_path, train=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

    estimator = ParaEst().to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.learning_rate)
    criteria = L2Loss(args.R, args.eta, device)

    lr_updated = 0
    best_avg_loss = np.inf
    best_epoch = 0
    avg_total_loss = np.zeros((args.epoch_num,), dtype=float)
    for epoch in tqdm(range(args.epoch_num)):
        estimator.train()
        if epoch // args.lr_update > lr_updated:
            lr_updated += 1
            adjust_learning_rate(optimizer, args.learning_rate * (0.5 ** lr_updated))
        for step, (img_ny, img_gt, alpha) in enumerate(train_loader):
            est = estimator(img_ny.permute(0,3,1,2))
            optimizer.zero_grad()
            loss = criteria(est, img_gt, img_gt, alpha)
            loss.backward()
            optimizer.step()
        avg_total_loss[epoch] = evaluateDataset(args, criteria, estimator, test_loader, len(dataset_test))
        print(loss, avg_total_loss[epoch]) #!
        if avg_total_loss[epoch] < best_avg_loss:
            best_avg_loss = avg_total_loss[epoch]
            torch.save(estimator, '%sbest_ran.pth'%args.data_path)
            best_epoch = epoch
    showCurve(args, avg_total_loss, 'loss_curve')
    print('-- Best epoch is {}, with average loss of {}'.format(best_epoch, best_avg_loss))