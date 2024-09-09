import torch, random, warnings, argparse
import numpy as np
warnings.filterwarnings("ignore") #忽略執行中的警告通知，讓輸出變更乾淨


def GetMainParameters():
    parser = argparse.ArgumentParser() #命令行參數解析器，讓使用者能夠通過命令行設置各種參數。
    # DATA
    parser.add_argument('--AnnotFile', type=str, default='./data/viton_annotations.pkl')    #設定數據集的位置和相關的文件路徑
    parser.add_argument('--DatasetRoot', type=str, default='D:/VITON/viton_plus')
    parser.add_argument('--SaveFolder_GMM', type=str, default='./results/image/GMM')
    parser.add_argument('--SaveFolder_GEN', type=str, default='./results/image/GEN')
    parser.add_argument('--RootCheckpoint_GMM', type=str, default='./results/checkpoint/GMM')
    parser.add_argument('--RootCheckpoint_GEN', type=str, default='./results/checkpoint/GEN')
    parser.add_argument('--SplitRatio', type=float, default=0.1)
    parser.add_argument('--NumWorkers', type=int, default=8)
    parser.add_argument('--BatchSize', type=int, default=8) #訓練批次的資料數量
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--num_head", type=int, default=4)
    # TRAINING        
    parser.add_argument('--Optim', type=str, default="adamw")   #優化器
    parser.add_argument('--Scheduler', type=str, default="")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--Seed', type=int, default=123)    #隨機種子
    parser.add_argument('--cuda', type=bool, default=True)
    return parser.parse_args([])    #解析上列參數，並返回一個包含所有參數的物件args


#設定損失權重參數
def GetLossWeightParameters(args):  
    # Regular G
    args.lambda_adv = 1.0
    args.lambda_skin_rm = 0.1
    return args


#設定優化器參數
def GetOptimParameters(args):   
    if args.Optim == 'adam':
        args.lr = 2e-4
        args.beta1 = 0.5
    elif args.Optim == 'adamw':
        args.lr = 5e-5
        args.beta1 = 0.5
    elif args.Optim == 'sgd':
        args.lr = 2e-4
        args.momentum = 0.9
    if args.Scheduler == 'cosine':
        args.T_max = 8
        args.T_mult = 2
    return args


#初始化評估指標
def MetricsInit(args):  
    args.best_loss = float('inf')   #最佳損失為正無窮，用來追蹤和比較模型訓練過程中表現的指標
    return args


#設定隨機種子
def SetupSeed(args):    
    #random、numpy、torch設置隨機種子，每次執行結果一致，便於重現實驗結果
    random.seed(args.Seed)
    np.random.seed(args.Seed)
    torch.cuda.manual_seed_all(args.Seed)


#設定模型參數
def GetModelSetting(args):  
    args.channels = [64, 128, 256]  #設定模型的通道數，代表不同層的特徵通道數，這可能會影響模型的複雜度和表現
    return args


#整合設定流程
def GetConfig():
    #依序呼叫上述函式，完成所有參數的初始化和設定
    args = GetMainParameters()
    args = GetLossWeightParameters(args)
    args = GetOptimParameters(args)
    args = MetricsInit(args)
    args = GetModelSetting(args)
    print('cuda') if torch.cuda.is_available() else print('cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #判斷是否有安裝cuda，有就用GPU，無則使用cpu
    return args
