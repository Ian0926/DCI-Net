import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--seed", type=int, default=666, help="random seed")
        self.parser.add_argument("--resume", action='store_true', help="if specified, resume the training")
        self.parser.add_argument("--results_dir", type=str, default='../results', help="path of saving models, images, log files")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/5 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        self.parser.add_argument("--train_bs_per_gpu", type=int, default=4, help="size of the training batches")
        self.parser.add_argument("--val_bs", type=int, default=1, help="size of the validating batches")
        self.parser.add_argument("--cropx", type=int, default=128, help="image size after cropping")
        self.parser.add_argument("--cropy", type=int, default=128, help="image size after cropping")
        self.parser.add_argument("--num_workers", type=int, default=6, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/5 : model defining... ------------------------------------------------
        self.parser.add_argument("--pretrained", type=str, default=None, help="pretrained model path")
        
        # ---------------------------------------- step 4/5 : requisites defining... ------------------------------------------------
        self.parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
        self.parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
        self.parser.add_argument("--scheduler", type=list, default=[500, 1000, 1500, 2000], help="learning rete scheduler")
        
        # ---------------------------------------- step 5/5 : training... ------------------------------------------------
        self.parser.add_argument("--print_gap", type=int, default=10, help="the gap between two print operations, in iteration")
        self.parser.add_argument("--val_gap", type=int, default=1, help="the gap between two validations, also the gap between two saving operation, in epoch")
    
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--outputs_dir", type=str, default='../outputs', help="path of saving images")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/4 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        
        # ---------------------------------------- step 3/4 : model defining... ------------------------------------------------
        self.parser.add_argument("--model_path", type=str, default=None, required=True, help="pretrained model path")
        
        # ---------------------------------------- step 4/4 : testing... ------------------------------------------------
        self.parser.add_argument("--save_image", action='store_true', help="if specified, save image when testing")
        
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')
        