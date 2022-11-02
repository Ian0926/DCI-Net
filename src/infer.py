import time
import torch
import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch import nn

from utils import *
from options import TestOptions
from models import PUNet
from datasets import SingleImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer/img'
# clean_dir(image_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('inferring data loading...')
infer_dataset = SingleImgDataset(data_source=opt.data_source)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{}'.format(len(infer_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = nn.DataParallel(PUNet()).cuda()
# print(opt.model_path)
# print(os.getcwd())
model.load_state_dict(torch.load(opt.model_path))
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()
    dataset = opt.data_source.split('/')[-1]

    check_path(image_dir + '/' + dataset)
    
    for i, (img_l, img_r, name_l, name_r) in enumerate(infer_dataloader):
        [img_l, img_r] = [x.cuda() for x in [img_l, img_r]]
        h, w = img_l.size(2), img_r.size(3)
        [img_l, img_r] = [check_padding(x) for x in [img_l, img_r]]
        [name_l, name_r] = [str(x[0]) for x in [name_l, name_r]]

        with torch.no_grad():
            start_time = time.time()
            pred_l, pred_r = model(torch.cat([img_l, img_r],1))
            times = time.time() - start_time
        
        [pred_l, pred_r] = [torch.clamp(x, 0, 1) for x in [pred_l, pred_r]]
        [pred_l, pred_r] = [x[:, :, :h, :w] for x in [pred_l, pred_r]]
        time_meter.update(times, 1)

        print('Iteration: ' + str(i+1) + '/' + str(len(infer_dataset)) + '  Processing image... ' + name_l + '  Time ' + str(times))
        
        if opt.save_image:
            save_image(pred_l, image_dir + '/' + dataset + '/' + name_l)
            save_image(pred_r, image_dir + '/' + dataset + '/' + name_r)
            
    print('Avg time: ' + str(time_meter.average()))

if __name__ == '__main__':
    main()
    