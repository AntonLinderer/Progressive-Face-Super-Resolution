import numpy as np
from PIL import Image
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pylab import plt
from face_alignment.models import FAN
from face_alignment.utils import *
from fan_model import fan_squeeze
from dataloader import *
from torch.nn import functional as F
import pdb

parser = argparse.ArgumentParser(description='FAN teacher student')
parser.add_argument('--batch-size', type=int, default=16,)
parser.add_argument('--max-iter', type=int, default=100000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save-path', type=str, default='./checkpoints/')
args = parser.parse_args()


dataset = CelebDataSet(state='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#model = torch.load('../Model_land/our_simp_c_model_003000.pth')
model = fan_squeeze().cuda()
#model = fan_squeeze4().cuda()

#TODO optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr)

def inf_train_gen(data_loader):
    while True:
        for _, dat in enumerate(data_loader):
            yield dat


face_alignment_net = FAN(4)
fan_weights = torch.load('./2DFAN4-11f355bf06.pth.tar')
face_alignment_net.load_state_dict(fan_weights)
face_alignment_net.cuda()

gen_load = inf_train_gen(dataloader)
iteration = 0
for i in range(args.max_iter):
    avg_loss = 0
    avg_loss_128 = 0
    for b, (img_d1, x4_target_image, img_d3, img_d4) in enumerate(dataloader):#x2_target_image, x4_target_image, x8_target_image, input_image
        real_image = x4_target_image.cuda() #64x64
        out = model(0.5*real_image+0.5)
        target = face_alignment_net(F.upsample(0.5*real_image+0.5, scale_factor=4, mode='bilinear'))[-1]
        loss = criterion(out, target)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = img_d4.size(0)
        if b%100==0:
            pts_f, pts_img_f = get_preds_fromhm(out)
            pts_f, pts_img_f = pts_f.view(-1, 68, 2) * (4/4), pts_img_f.view(-1, 68, 2)
            pts_r, pts_img_r = get_preds_fromhm(target)
            pts_r, pts_img_r = pts_r.view(-1, 68, 2) * (4/4), pts_img_r.view(-1, 68, 2)
            for bch in range(batch_size):
                for pt in range(pts_r.size(1)):
                    x = pts_r[bch, pt, 1].long()
                    y = pts_r[bch, pt, 0].long()
                    real_image[bch, 0, x-1:x+1, y-1:y+1] = 1.0
                    real_image[bch, 1, x-1:x+1, y-1:y+1] = -1.0
                    real_image[bch, 2, x-1:x+1, y-1:y+1] = -1.0
                for pt in range(pts_f.size(1)):
                    x_f = pts_f[bch, pt, 1].long()
                    y_f = pts_f[bch, pt, 0].long()
                    real_image[bch, 0, x_f-1:x_f+1, y_f-1:y_f+1] = -1.0
                    real_image[bch, 1, x_f-1:x_f+1, y_f-1:y_f+1] = -1.0
                    real_image[bch, 2, x_f-1:x_f+1, y_f-1:y_f+1] = 1.0
            f_imgs = utils.make_grid((real_image.data*0.5+0.5))
            utils.save_image(f_imgs,'./result/align_img.jpg')
            print('batch_loss %.4f'%loss.item())
        iteration+=1
        
        if iteration%200==0:
            torch.save(model, args.save_path+'compressed_model_'+str(iteration).zfill(6)+'.pth')
            print(iteration, 'model saved')
