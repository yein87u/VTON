from torchvision.utils import save_image, make_grid
import os, gc, torch, shutil, cv2, glob
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from einops import repeat


def resume_model(args, model_G, model_D):
    files = glob.glob(os.path.join(args.RootCheckpoint, '*.pth.tar'))
    if len(files) == 0: return 0, model_G, model_D   
    # 找出loss最低的權重
    filenames = [os.path.basename(file) for file in files]
    digit_map = {float(filename.split('loss_')[-1].split('_ckpt.best.pth.tar')[0]):filename for filename in filenames}
    # 取得權重
    weight = torch.load(os.path.join(args.RootCheckpoint, digit_map[min(digit_map)]))
    # 匯入權重
    start_epoch = weight['args'].epoch
    model_G.load_state_dict(weight['model_G_state_dict'])
    model_D.load_state_dict(weight['model_D_state_dict'])
    return start_epoch, model_G, model_D

#對給定的神經網絡模型進行權重初始化
def weights_initialization(net, init_type='normal', gain=0.02):
    def _weights_init(m):
        classname = m.__class__.__name__
        #檢查是否有weight屬性，檢查是否為卷積層或全連階層
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            #根據type選擇初始化方法
            if init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            #若該層設有偏差則將偏差設為0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:   #檢查當前層是否為BatchNorm2d
            init.normal_(m.weight.data, 1.0, gain)  #權重張量，正態分佈初始化，均值為1.0，標準差為gain
            init.constant_(m.bias.data, 0.0)        #偏差張量，將張量中所有偏差值(bias)設為0.0
    print('initialize network with %s' % init_type) #輸出正在使用的初始化方法
    net.apply(_weights_init)    #將_weights_init運用在模型的所有子模塊(層)上

#清空python、cuda記憶體，釋放空間
def flush():
    gc.collect()    #垃圾回收模組
    torch.cuda.empty_cache()    #釋放GPU空間


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Visualizer():
    def __init__(self, args, SaveFolder_root):
        self.args = args
        self.SaveFolder = SaveFolder_root
        os.makedirs(self.SaveFolder, exist_ok=True) #確認檔案是否存在
                
    
    def plot_loss(self, loss_history):
        losses = []; names = []
        for k, v in loss_history.items():   #將歷史損失存入陣列
            names.append(k); losses.append(v)
        #畫圖函式
        self.plot_curves(self.SaveFolder, 'loss', losses, names, ylabel='Loss')

    #畫圖，並指定保存圖表之路徑，y軸顯示損失值
    def plot_curves(self, path, name, point_list, curve_names=None, freq=1, xlabel='Epoch',ylabel=None):
        if curve_names is None:
            curve_names = [''] * len(point_list)
        else:
            assert len(point_list) == len(curve_names)  #檢查確保每條線都有相對應的名稱

        x = (np.arange(len(point_list[0])) + 1) * freq  #
        if len(point_list) <= 10:
            cmap = plt.get_cmap('tab10')
        else:
            cmap = plt.get_cmap('tab20')
        for i, (point, curve_name) in enumerate(zip(point_list, curve_names)):
            assert len(point) == len(x)
            plt.plot(x, point, color=cmap(i), label=curve_name)
            
        if ylabel is not None: plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.savefig(os.path.join(path, name + '.png'))
        plt.close()
    
    def vis_train(self, img_train_batch, order):
        for idx, img in enumerate(img_train_batch):
            n, c, h, w = img.shape
            if c == 1:
                img_train_batch[idx] = repeat(img, 'n c h w -> n (repeat c) h w', repeat=3)
        img_train_batch = torch.cat(img_train_batch, dim=3)
        save_path = os.path.join(self.SaveFolder, f'epoch_{self.args.epoch}_fake_{order}.png')
        ndarr = make_grid(img_train_batch, 1).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        cv2.imwrite(save_path, ndarr)

def save_checkpoint(checkpoint_root, state, is_best):
    os.makedirs(checkpoint_root, exist_ok=True)
    filename = os.path.join(checkpoint_root, f"checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best: 
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
        print('best model saved.')