# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/extract_feature_v1.py

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from .backbone import Backbone
from tqdm import tqdm
from PIL import Image


def get_embeddings(data_root, model_root, input_size=[112, 112], embedding_size=512):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check data and model paths
    assert os.path.exists(data_root)
    assert os.path.exists(model_root)
    print(f"Data root: {data_root}")

    # define image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )

    # define data loader
    dataset = datasets.ImageFolder(data_root, transform)
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
    )
    print(f"Number of classes: {len(loader.dataset.classes)}")

    # load backbone weigths from a checkpoint
    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()

    # get embedding for each face
    embeddings = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        for idx, (image, _) in enumerate(
            tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
        ):
            embeddings[idx, :] = F.normalize(backbone(image.to(device))).cpu()

    # get all original images
    images = []
    for img_path, _ in dataset.samples:
        img = cv2.imread(img_path)
        images.append(img)

    return images, embeddings


#get embeddings but for only one image
def get_single_embedding_from_file(img_path, model_root, input_size=[112, 112], embedding_size=512):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check image and model paths
    assert os.path.exists(img_path)
    assert os.path.exists(model_root)
    print(f"Image path: {img_path}")

    # define image preprocessing
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load the image
    img = Image.open(img_path).convert('RGB')
    tensor_img = transform(img).unsqueeze(0)  # Add a batch dimension

    # load backbone weights from a checkpoint
    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()

    # get embedding for the image
    with torch.no_grad():
        embedding = F.normalize(backbone(tensor_img.to(device))).cpu().numpy()

    # Load the image using cv2 for visualization purposes
    img_cv2 = cv2.imread(img_path)

    return img_cv2, embedding


#get embeddings but for only one image
def get_single_embedding_from_memory(img, model_root, input_size=[112, 112], embedding_size=512):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert os.path.exists(model_root)

    # define image preprocessing
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load the image
    tensor_img = transform(img).unsqueeze(0)  # Add a batch dimension

    # load backbone weights from a checkpoint
    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()

    # get embedding for the image
    with torch.no_grad():
        embedding = F.normalize(backbone(tensor_img.to(device))).cpu().numpy()


    return embedding