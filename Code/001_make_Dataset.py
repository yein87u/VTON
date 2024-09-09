import sys
import pickle
import config
import os
import os.path as osp
from tqdm import tqdm
sys.path.append('.')


class AnnotationBuilding():
    def __init__(self, args):   #當Class被呼叫，直接執行，並將args存取在物件self的參數args中
        self.args = args

    def CreateAnnotationFile(self):
        #標註文件函數
        def pickup_data(phase):
            # load data list
            samples = []
            with open(osp.join(self.args.DatasetRoot, f"{phase}_pairs.txt"), 'r') as f:
                for line in tqdm(f.readlines()):
                    im_name, c_name = line.strip().split()
                    im_name_png = im_name.replace('.jpg', '.png')
                    pose_name = im_name.replace('.jpg', '_keypoints.json')
                    
                    #檢查檔案是否存在
                    pose_file = osp.join(self.args.DatasetRoot, phase, 'pose', pose_name)
                    densepose_file = osp.join(self.args.DatasetRoot, phase, 'densepose', im_name_png)
                    parse_file = osp.join(self.args.DatasetRoot, phase, 'image-parse-new', im_name_png)
                    image_file = osp.join(self.args.DatasetRoot, phase, 'image', im_name)
                    cloth_file = osp.join(self.args.DatasetRoot, phase, 'cloth', c_name)
                    cloth_mask_file = osp.join(self.args.DatasetRoot, phase, 'cloth-mask', c_name)
                    
                    if not os.path.isfile(pose_file):
                        continue
                    # if not os.path.isfile(densepose_file):
                    #     continue
                    if not os.path.isfile(parse_file):
                        continue
                    if not os.path.isfile(image_file):
                        continue
                    if not os.path.isfile(cloth_file):
                        continue
                    if not os.path.isfile(cloth_mask_file):
                        continue
                    
                    #收集有效資料樣本，字典
                    samples.append({
                        'cloth_name': c_name, 
                        'image_name': im_name,
                        'cloth'     : cloth_file, 
                        'cloth_mask': cloth_mask_file,
                        'image'     : image_file,
                        'image_parse': parse_file,
                        'pose_label': pose_file,
                        'densepose_label': densepose_file,
                    })
            return samples
        
        #保存標註資料
        training_sample = pickup_data('train')
        testing_sample = pickup_data('test')
        data_file = {'Training_Set':training_sample, 'Testing_Set':testing_sample,}
        pickle.dump(data_file, open(self.args.AnnotFile, 'wb')) #pickle序列化，二進位寫入檔案
        print(training_sample[0])
        print(testing_sample[0])

if __name__== '__main__':
    annotationBuilding = AnnotationBuilding(config.GetConfig()) #執行config.py初始化，並回傳args
    annotationBuilding.CreateAnnotationFile()

