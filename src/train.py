import time
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from torch import nn

from utils import *
from options import TrainOptions
from models import PUNet
from losses import LossL1, LossFreqReco, LossSSIM, LossTV, LossEdge
from datasets import PairedImgDataset

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

gpu_num = torch.cuda.device_count()

set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = PairedImgDataset(data_source=opt.data_source, mode='train', crop=[opt.cropx, opt.cropy], random_resize=None)
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs_per_gpu*gpu_num, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs_per_gpu*gpu_num))

print('validating data loading...')
val_dataset = PairedImgDataset(data_source=opt.data_source, mode='val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = nn.DataParallel(PUNet()).cuda()
print_para_num(model)

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_tv = LossTV().cuda()
criterion_fre = LossFreqReco().cuda()

if opt.resume:
    state = torch.load(models_dir + '/latest.pth')
    optimizer = state['optimizer']
    scheduler = state['scheduler']
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler, 0.5)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    
    optimal = [0., 0.]
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        
        print('Resume from epoch %d' % (start_epoch), optimal)
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch, optimal)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch, optimal)
        
    writer.close()
    
def train(epoch, optimal):
    model.train()
    
    max_iter = len(train_dataloader)
        
    iter_ssim_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(train_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        cur_batch = imgs_l.shape[0]
        
        optimizer.zero_grad()
        input = torch.cat([imgs_l, imgs_r], 1)
        preds_l, preds_r = model(input)

        loss_tv = criterion_tv(preds_l) + criterion_tv(preds_r)
        loss_fre = criterion_fre(preds_l, gts_l) + criterion_fre(preds_r, gts_r)

        loss = 0.1*loss_tv + loss_fre
        
        loss.backward()
        optimizer.step()
        
        iter_ssim_meter.update(loss.item()*cur_batch, cur_batch)
             
        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Best: {:.4f}/{:.4f} loss: {:.4f} Time: {:.4f} LR: {:.8f}'.format(epoch, 
            opt.n_epochs, i + 1, max_iter, optimal[0], optimal[1], iter_ssim_meter.average(), iter_timer.timeit(), scheduler.get_last_lr()[0]))
            writer.add_scalar('Loss_cont', iter_ssim_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            
            
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'optimal': optimal, 'optimizer': optimizer, 'scheduler': scheduler}, models_dir + '/latest.pth')
    scheduler.step()
    
def val(epoch, optimal):
    model.eval()
    
    print(''); print('Validating...', end=' ')
    
    psnr_meter_l = AverageMeter()
    ssim_meter_l = AverageMeter()
    psnr_meter_r = AverageMeter()
    ssim_meter_r = AverageMeter()
    timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(val_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        h, w = gts_l.size(2), gts_l.size(3)
        [imgs_l, imgs_r] = [check_padding(x) for x in [imgs_l, imgs_r]]
        input = torch.cat([imgs_l, imgs_r], 1)

        with torch.no_grad():
            preds_l, preds_r = model(input)
        [preds_l, preds_r] = [x[:, :, :h, :w] for x in [preds_l, preds_r]]

        psnr_value_l, ssim_value_l = get_metrics(preds_l, gts_l, psnr_only=False)
        psnr_meter_l.update(psnr_value_l, imgs_l.shape[0])
        ssim_meter_l.update(ssim_value_l, imgs_l.shape[0])

        psnr_value_r, ssim_value_r = get_metrics(preds_r, gts_r, psnr_only=False)
        psnr_meter_r.update(psnr_value_r, imgs_l.shape[0])
        ssim_meter_r.update(ssim_value_r, imgs_l.shape[0])
    
    psrn_cur = (psnr_meter_l.average() + psnr_meter_r.average()) / 2
    if (optimal[0] + optimal[1]) / 2 < psrn_cur:
        optimal[0] = psnr_meter_l.average()
        optimal[1] = psnr_meter_r.average()
        torch.save(model.state_dict(), models_dir + '/optimal.pth')

    print('Epoch[{:0>4}/{:0>4}] PSNR/SSIM (left,right): {:.4f}/{:.4f},{:.4f}/{:.4f} Best: {:.4f}/{:.4f} Time: {:.4f}'.format(
        epoch, opt.n_epochs, psnr_meter_l.average(), ssim_meter_l.average(), psnr_meter_r.average(), ssim_meter_r.average(), 
        optimal[0], optimal[1],timer.timeit())); print('')
    
if __name__ == '__main__':
    main()
    