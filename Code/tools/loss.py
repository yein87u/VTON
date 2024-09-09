import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange


def norm(x):
    return x * 2 - 1

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Vgg19(nn.Module): #nn.Module是神經網絡模型的基礎類
    def __init__(self, requires_grad=False):    #requires_grad 控制網路中的參數是否要計算梯度
        super().__init__()  #繼承nn.Module父類，並不是重跑一次
        vgg_pretrained_features = models.vgg19(pretrained=True).features    #從模組裡面載入Vgg19模型，並提取特徵層

        #nn.Sequential構建神經網路的容器，允許將多個神經網路層按順序組合在一起。下列共有5個切片
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        #將特徵層分配給各個容器中，並且指定名稱
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        #凍結模型參數
        if not requires_grad:
            for param in self.parameters(): #返回模型中的所有參數
                param.requires_grad = False #在訓練過程中，這些參數將不會進行反向傳播計算，也不會被更新。

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        return content_loss, style_loss
    

def compute_gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G


class LossOperator():
    def __init__(self, args):
        self.args = args
        #設置標籤，創建判別器時，需定義樣本的標籤
        self.real_label = torch.tensor(1.0) #建立張量，此類型是torch.float32
        self.fake_label = torch.tensor(0.0) 
        
        self.criterion_L1 = torch.nn.L1Loss().to(args.device)   #實體化損失函數，平均絕對誤差(MAE)，將損失函式移動到GPU上運算
        self.criterion_L2 = torch.nn.MSELoss().to(args.device)  #實體化損失函數，均方誤差（MSE），將損失函式移動到GPU上運算
        
        self.vgg = Vgg19().to(self.args.device) #實體化Vgg19模型，並移動到GPU上運算
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  #設定Vgg各層的權重參數

        #濾波器銓重初始化
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]  #水平
        
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]  #垂直
        
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]  #對角
        
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]  #對角
        
        weight_array = np.ones([3, 3, 1, 4])    #建立大小為 3x3x1x4 的權重張量

        #將定義的濾波器插入到權重張量中，使得濾波器在張量的不同通道中
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        #調整為GPU浮點張量，並且使用permute修改維度的順序，直接指定某維度修改至某維度
        weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        #設置為不需要計算梯度的參數，故這些權重在訓練過程中將保持不變
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

    #計算Vgg19的損失
    def calc_vgg_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y) #將x, y輸入Vgg模型，並取得x, y經過Vgg各層後的特徵
        content_loss = 0.0  #內容損失
        style_loss = 0.0    #風格損失
        for i in list(range(len(x_vgg))):   #遍歷x_vgg中每一張特徵圖
            #計算內容損失，並使用平均絕對誤差(MAE)計算，.detach()是為了讓y的梯度不會被計算，避免反向傳播
            content_loss += self.vgg_weights[i] * self.criterion_L1(x_vgg[i], y_vgg[i].detach())
            #計算風格損失，compute_gram用來計算特徵圖的 Gram 矩陣
            style_loss += self.criterion_L1(compute_gram(x_vgg[i]), compute_gram(y_vgg[i].detach()))
        return content_loss, style_loss
    
    #計算張量(mask)的總變異損失，其用於圖像生成、處理、增強的正則化技術，增加平滑，降低雜訊、不自然現象
    def calc_total_variation_loss(self, mask):
        tv_h = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        tv_w = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))
    
    def calc_laplacian_loss(self, flow):
        flow_x, flow_y = torch.split(flow, 1, dim=1)
        delta_x = F.conv2d(flow_x, self.weight)
        delta_y = F.conv2d(flow_y, self.weight)

        b, c, h, w = delta_x.shape
        image_elements = b * c * h * w

        loss_flow_x = (delta_x.pow(2)+ self.args.epsilon ** 2).pow(0.45)
        loss_flow_y = (delta_y.pow(2)+ self.args.epsilon ** 2).pow(0.45)

        loss_flow_x = torch.sum(loss_flow_x) / image_elements
        loss_flow_y = torch.sum(loss_flow_y) / image_elements
        return loss_flow_x + loss_flow_y