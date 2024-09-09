import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_fid import fid_score

mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


def main(real_images_folder, fake_images_folder):
    # Set up dtype
    device = torch.device("cuda:0")  # you can change the index of cuda

    # Load inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
    preds = np.zeros((N, 1000))

    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    main()