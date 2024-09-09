import cv2
import sys
import json
import torch
import pickle
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
sys.path.append('.')

semantic_densepose_labels = [
    [0, 0, 0],
	[105, 105, 105],
	[85, 107, 47],
	[139, 69, 19],
	[72, 61, 139],
	[0, 128, 0],
	[154, 205, 50],
	[0, 0, 139],
	[255, 69, 0],
	[255, 165, 0],
	[255, 255, 0],
	[0, 255, 0],
	[186, 85, 211],
	[0, 255, 127],
	[220, 20, 60],
	[0, 191, 255],
	[0, 0, 255],
	[216, 191, 216],
	[255, 0, 255],
	[30, 144, 255],
	[219, 112, 147],
	[240, 230, 140],
	[255, 20, 147],
	[255, 160, 122],
	[127, 255, 212]
]

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, args, phase='train'):
        self.args = args
        self.phase = phase
        self.fine_width  = self.args.fine_width
        self.fine_height = self.args.fine_height
        self._read_path_label()
        self._setup_transforms()
        self.image_grid = cv2.imread('./grid.png')
        self.image_grid = self.transform_bin(image=self.image_grid)["image"]
        self.image_grid = self.from_255_to_norm(self.image_grid)


    def _read_path_label(self):
        assert self.phase in ['train', 'test'], 'Phase must be in : Train or Test'
        pkl = pickle.load(open(self.args.AnnotFile, 'rb'))
        if self.phase == 'train':
            self.data = pkl['Training_Set']
        elif self.phase == 'test':
            self.data = pkl['Testing_Set']
        self.dataset_size = len(self.data)

    def _setup_transforms(self):
        self.transform_bin = A.Compose([
            A.Resize(self.fine_height, self.fine_width), 
            ToTensorV2()
        ])
            
    def load_keypoints(self, pose_file):
        with open(pose_file, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        return pose_data.astype(np.float32)
    
    def get_parsing_part(self, parse_mask):
        parse_cloth  = (parse_mask==5).astype(np.float32)     
        parse_arm    = (parse_mask==14).astype(np.float32) + (parse_mask==15).astype(np.float32)
        parse_face   = (parse_mask==2 ).astype(np.float32) + (parse_mask==13).astype(np.float32)
        parse_others = (parse_mask==6 ).astype(np.float32) + (parse_mask==8 ).astype(np.float32) + \
                       (parse_mask==9 ).astype(np.float32) + (parse_mask==10 ).astype(np.float32) + \
                       (parse_mask==12).astype(np.float32) + (parse_mask==16).astype(np.float32) + \
                       (parse_mask==17).astype(np.float32) + (parse_mask==18).astype(np.float32) + \
                       (parse_mask==19).astype(np.float32) 

        parse_arm    = torch.from_numpy(parse_arm).unsqueeze(0)
        parse_face   = torch.from_numpy(parse_face).unsqueeze(0)
        parse_others = torch.from_numpy(parse_others).unsqueeze(0) 
        return parse_arm, parse_face, parse_cloth, parse_others
    
    def load_densepose(self, path, size):
        dense_mask = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        dense_mask = cv2.resize(dense_mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        densepose = np.zeros((25, size[0], size[1]))
        densepose_fore = np.zeros((size[0], size[1]))
        for i, color in enumerate(semantic_densepose_labels):
            densepose[i] = np.all(dense_mask==color, axis=-1)
            densepose_fore[np.all(dense_mask==color, axis=-1)] = i
        hand_mask = torch.FloatTensor(densepose_fore==3) + torch.FloatTensor(densepose_fore==4)
        densepose = torch.FloatTensor(densepose)
        densepose_fore = torch.FloatTensor(densepose_fore / 24).unsqueeze(0)
        return densepose, densepose_fore, hand_mask

    def draw_image(self, pose_data):
        r = self.args.radius
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        image_pose = np.zeros((self.fine_height, self.fine_width))
        for i in range(point_num):
            one_map = np.zeros((self.fine_height, self.fine_width)).astype(np.float32)
            pointx = int(pose_data[i, 0])
            pointy = int(pose_data[i, 1])
            if pointx > 1 and pointy > 1:
                cv2.rectangle(one_map, (pointx-r, pointy-r), (pointx+r, pointy+r), (255, 255, 255), -1)
                cv2.rectangle(image_pose, (pointx-r, pointy-r), (pointx+r, pointy+r), (255, 255, 255), -1)
            one_map = self.transform_bin(image=one_map)["image"]                      # [-1, 1]
            one_map = self.from_255_to_norm(one_map)
            pose_map[i] = one_map[0]
        image_pose = self.transform_bin(image=image_pose)["image"]                      # [-1, 1]
        image_pose = self.from_255_to_norm(image_pose)
        return image_pose, pose_map
    
    def from_255_to_norm(self, x):
        return x / 255 * 2 - 1
    
    def to_norm(self, x):
        return x * 2 - 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # densepose
        # image-parse
        image       = cv2.imread(self.data[idx]['image'])
        cloth       = cv2.imread(self.data[idx]['cloth'])
        cloth_mask  = cv2.imread(self.data[idx]['cloth_mask'], 0)
        image_parse = cv2.imread(self.data[idx]['image_parse'], 0) 
        pose_data   = self.load_keypoints(self.data[idx]['pose_label'])
        image_pose, pose_map = self.draw_image(pose_data)       # [-1, 1]    # torch.tensor

        cloth_mask = (cloth_mask > 128).astype(np.float32)                  # [0, 1]
        cloth = cloth * np.expand_dims(cloth_mask, -1)                      # [0, 255]

        parse_arm_mask, parse_face_mask, person_cloth_mask, parse_others_mask = self.get_parsing_part(image_parse)                             # [0, 1]
        person_clothes = image * np.expand_dims(person_cloth_mask, -1)    # [0, 255]
        preserve_mask = torch.cat([parse_face_mask, parse_others_mask], dim=0)                                              # [0, 1]    

        # densepose, densepose_fore, hand_mask = self.load_densepose(self.data[idx]['densepose_label'],  image_parse.shape)   # [0, 1]    

        image = self.transform_bin(image=image)["image"]                            # [0, 255]
        cloth = self.transform_bin(image=cloth)["image"]                            # [0, 255]
        person_clothes = self.transform_bin(image=person_clothes)["image"]          # [0, 255]

        face_image = image * parse_face_mask                        # [0, 255]
        # hand_image = image * parse_arm_mask * hand_mask           # [0, 255]
        other_clothes_image = image * parse_others_mask             # [0, 255]
        # agnostic = face_image + other_clothes_image + hand_image  # [0, 255]
        agnostic = face_image + other_clothes_image                 # [0, 255]
        
        cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)                      # [0, 1]
        person_cloth_mask = torch.from_numpy(person_cloth_mask).unsqueeze(0)    # [0, 1]

        image = self.from_255_to_norm(image)                        # [-1, 1]
        cloth = self.from_255_to_norm(cloth)                        # [-1, 1]
        person_clothes = self.from_255_to_norm(person_clothes)      # [-1, 1]
        agnostic = self.from_255_to_norm(agnostic)                  # [-1, 1]
        preserve_mask = self.to_norm(preserve_mask)                 # [-1, 1]
        # densepose = self.to_norm(densepose)                         # [-1, 1]
        cloth_mask = self.to_norm(cloth_mask)                       # [-1, 1]
        person_cloth_mask = self.to_norm(person_cloth_mask)     # [-1, 1]

        # person_shape  = torch.cat([agnostic, preserve_mask, densepose, pose_map], dim=0)       # [-1, 1]   
        person_shape  = torch.cat([agnostic, preserve_mask, pose_map], dim=0)       # [-1, 1]   

        return {
            'cloth_name': self.data[idx]['cloth_name'], # for visualization
            'image_name': self.data[idx]['image_name'], # for visualization or ground truth
            'image'     : image,                        # (192, 256, 3 ) for visualization 
            'cloth'     : cloth,                        # (192, 256, 3 ) for input         
            'cloth_mask': cloth_mask,                   # (192, 256, 1 ) for input         
            'agnostic'  : agnostic,                     # (192, 256, 22) for input         
            'person_shape'  : person_shape,             # (192, 256, 22) for input         
            'person_clothes': person_clothes,           # (192, 256, 3 ) for ground truth  
            'person_clothes_mask': person_cloth_mask, # (192, 256, 1 ) for ground truth  
            # 'densepose': densepose,                     # (192, 256, 1 ) for ground truth  
            # 'densepose_fore': densepose_fore,           # (192, 256, 1 ) for ground truth  
            'image_pose': image_pose,                   # (192, 256, 1 ) for visualization 
            'image_grid': self.image_grid,              # (192, 256, 3 ) for visualization
        }
        
    
class data_prefetcher():
    def __init__(self, args, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(args.device)
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label 
        self.preload()
        return image, label