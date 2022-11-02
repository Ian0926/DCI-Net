import time
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import PUNet
from datasets import PairedImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

# image_dir = opt.outputs_dir + '/' + opt.experiment + '/test/img'
# clean_dir(image_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('testing data loading...')
test_dataset = PairedImgDataset(data_source=opt.data_source, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{}'.format(len(test_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = nn.DataParallel(PUNet()).cuda()
if 'latest' in opt.model_path:
    model.load_state_dict(torch.load(opt.model_path)['model'])
else:
    model.load_state_dict(torch.load(opt.model_path))
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter_l = AverageMeter()
    ssim_meter_l = AverageMeter()
    psnr_meter_r = AverageMeter()
    ssim_meter_r = AverageMeter()
    time_meter = AverageMeter()
    
    for i, (img_l, img_r, gt_l, gt_r) in enumerate(test_dataloader):
        [img_l, img_r, gt_l, gt_r] = [x.cuda() for x in [img_l, img_r, gt_l, gt_r]]
        h, w = img_l.size(2), img_r.size(3)
        [img_l, img_r] = [check_padding(x) for x in [img_l, img_r]]

        with torch.no_grad():
            start_time = time.time()
            pred_l, pred_r = model(torch.cat([img_l, img_r],1))
            times = time.time() - start_time
        
        [pred_l, pred_r] = [x[:, :, :h, :w] for x in [pred_l, pred_r]]

        psnr_value_l, ssim_value_l = get_metrics(pred_l, gt_l, psnr_only=False)
        psnr_value_r, ssim_value_r = get_metrics(pred_r, gt_r, psnr_only=False)

        psnr_meter_l.update(psnr_value_l, 1)
        ssim_meter_l.update(ssim_value_l, 1)
        psnr_meter_r.update(psnr_value_r, 1)
        ssim_meter_r.update(ssim_value_r, 1)

        time_meter.update(times, 1)

        print('Iteration: ' + str(i+1) + '/' + str(len(test_dataset)) + '  Time ' + str(times))
        
    psnr_l, ssim_l = psnr_meter_l.average(), ssim_meter_l.average()
    psnr_r, ssim_r = psnr_meter_r.average(), ssim_meter_r.average()
    psnr, ssim = (psnr_l + psnr_r) / 2, (ssim_l + ssim_r)/2
    print('Avg time: ' + str(time_meter.average()))
    print('PSNR/SSIM (left): ' + str(round(psnr_l, 2)) + ' ' + str(round(ssim_l, 3)))
    print('PSNR/SSIM(right): ' + str(round(psnr_r, 2)) + ' ' + str(round(ssim_r, 3)))
    print('PSNR/SSIM: ' + str(round(psnr, 2)) + ' ' + str(round(ssim, 3)))

if __name__ == '__main__':
    main()
    