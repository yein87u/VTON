import os
import cv2
import tqdm
import torch
import config
import datasets
import numpy as np
import skimage.metrics as skm
from models.model_gmm import CAFWM
from models.model_gen import ResUnetGenerator


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = skm.structural_similarity(
            reference_image, generated_image, 
            sigma=1.5,
            multichannel=True,
            gaussian_weights=True, 
            use_sample_covariance=False, 
            data_range=generated_image.max() - generated_image.min()
        )
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)



class ModelLearning():
    def __init__(self, args):
        self.args = args
        self.args = config.MetricsInit(self.args)
        self.GetDataloader()


    def GetDataloader(self):
        test_dataset = datasets.ImagesDataset(self.args, phase='test')
        self.test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=self.args.NumWorkers)
        self.args.len_dataset = len(self.test_dataLoader)


    def de_norm(self, x):
        return (x + 1) / 2 * 255.


    def generated_images(self):
        self.net_warp.eval()
        self.net_gen.eval()
        count = 0
        with tqdm.tqdm(self.test_dataLoader, desc="training") as pbar:
            for idx, sample in enumerate(pbar):
                image       = sample['image'].to(self.args.device)
                cloth       = sample['cloth'].to(self.args.device)
                cloth_mask  = sample['cloth_mask'].to(self.args.device)
                agnostic    = sample['agnostic'].to(self.args.device)
                person_shape= sample['person_shape'].to(self.args.device)
                person_clothes_mask = sample['person_clothes_mask'].to(self.args.device)

                with torch.set_grad_enabled(False):
                    # Warping Network
                    output = self.net_warp(cloth, cloth_mask, person_shape)
                    # Generative Network
                    warped_mask = output['warping_masks'][-1]
                    warped_cloth = output['warping_cloths'][-1]
                    gen_inputs = torch.cat([agnostic, warped_cloth, warped_mask], dim=1)
                    gen_outputs = self.net_gen(gen_inputs)

                    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                    p_rendered = torch.tanh(p_rendered)
                    m_composite = torch.sigmoid(m_composite)
                    m_composite1 = m_composite * warped_mask
                    m_composite =  person_clothes_mask * m_composite1
                    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

                for i in range(len(image)):
                    count = count + 1
                    print(torch.max(p_tryon))
                    print(torch.min(p_tryon))
                    cv2.imwrite(f'./results/for_evaluate/real/{count:06d}.png', self.de_norm(image[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/fake/{count:06d}.png', self.de_norm(p_tryon[i].permute(1, 2, 0).cpu().detach().numpy()))

    def test(self):
        weight = torch.load(os.path.join(self.args.RootCheckpoint_GEN, 'checkpoint.best.pth.tar'))
        self.net_warp = CAFWM(self.args).to(self.args.device)
        self.net_gen  = ResUnetGenerator(32, 4, 5, ngf=64).to(self.args.device)
        self.net_gen.load_state_dict(weight['GEN_state_dict'])
        self.net_warp.load_state_dict(weight['GMM_state_dict'])
        self.generated_images()


if __name__ == '__main__':
    modelLearning = ModelLearning(config.GetConfig())
    modelLearning.test()

