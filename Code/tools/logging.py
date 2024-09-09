import os, logging, time
# logging 日誌記錄模組

class Logger(): #在實體化過後self將會在methon中自動傳入
    def __init__(self, args, SaveFolder_root):
        self.args = args
        self.SaveFolder = SaveFolder_root   #存取路徑
        self.init_loss()    #初始化損失列表
        self.CreateLogger() #創建日誌物件，並且將日誌訊息寫入檔案或輸出至終端
        self.Reset_PerEpochLoss()   #重設與損失相關的暫時數據

    #初始化損失數值列表
    def init_loss(self):
        self.loss_list = [
            'wloss-content', 'wloss-mask', 'wloss-style', 'wloss-lapla', 'wloss-smooth', 'wloss-warp',  #神經網路權重有關之損失
            'gloss-vgg-bg', 'gloss-con-bg', 'gloss-vgg', 'gloss-con', 'gloss-mask', 'gloss-gen',    #生成有關之損失
            'loss-total',]  #總損失
        self.loss_history = {k:[] for k in self.loss_list}  #創建字典，鍵為損失列表中的名稱，紀錄每次訓練步驟中損失的數值

    #創建日誌物件，並且將日誌訊息寫入檔案或輸出至終端
    def CreateLogger(self, console=True):
        os.makedirs(self.SaveFolder, exist_ok=True) #檢查路徑是否存在
        _level = logging.INFO   #日誌級別，INFO忽略DEBUG級別的詳細資訊

        self.logger = logging.getLogger()   #取得日誌記錄器物件
        self.logger.setLevel(_level)    #設置級別

        #即時觀看
        if console:
            cs = logging.StreamHandler()    #實體化日誌處理器，將日誌訊息傳至控制台或終端
            cs.setLevel(_level) #設置級別
            self.logger.addHandler(cs)  #將處理器添加到self.logger中，使日誌訊息能輸出

        #寫入檔案
        file_name = os.path.join(self.SaveFolder, 'model_log.txt')  #合成路徑
        fh = logging.FileHandler(file_name, mode='w')   #將日誌訊息寫入檔案(覆寫)，若檔案不存在，將會自動新增檔案
        fh.setLevel(_level) #設置級別
        self.logger.addHandler(fh)  #寫進特定檔案當中，而不是控制台或終端

    #重設與損失相關的暫時數據
    def Reset_PerEpochLoss(self):
        self.start_time = time.time()   #紀錄當前時間為開始時間
        self.loss_tmp = {k:0.0 for k in self.loss_list} #創建字典，鍵為損失列表中的名稱，並賦予初始值0.0

    def Log_PerEpochLoss(self):
        log_str = '\n' + '='*40 + f'\nEpoch {self.args.epoch}, time {time.time() - self.start_time:.2f} s'
        self.logger.info(log_str) if self.logger is not None else print(log_str)
        for k, v in self.loss_tmp.items():
            self.loss_history[k].append(v)
            log_str = f'{k:s}\t{v:.6f}'
            self.logger.info(log_str) if self.logger is not None else print(log_str)  
        self.logger.info('='*40) if self.logger is not None else print('='*40)
    

    def print_network(self, model, name):
        num_params = 0  #用來計算累加模型參數的總數
        for p in model.parameters():    #將模型中的參數全部代出
            num_params += p.numel() #.numel()是代出的參數中擁有的參數數量
        log_str = f'{name:s}, the number of parameters: {num_params:d}' #格式化輸出結果
        #若日誌有被實體化，則紀錄訊息；若沒有配備日誌也可以被印出到終端
        self.logger.info(log_str) if self.logger is not None else print(log_str)    
