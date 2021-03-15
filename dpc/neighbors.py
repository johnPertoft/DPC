import os
import random
import sys
sys.path.append('../utils')

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from augmentation import Normalize
from augmentation import RandomSizedCrop
from augmentation import Scale
from augmentation import ToTensor
from dataset_3d import UCF101_3d
from model_3d import DPC_RNN


def main():
    st.title("Neighbor visualization")

    global cuda; cuda = torch.device("cuda")

    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.0),
        Scale(size=(128, 128)),
        ToTensor(),
        Normalize()
    ])

    dataset = UCF101_3d(
        mode="test",
        transform=transform,
        seq_len=5,
        num_seq=8,
        downsample=3)
    
    data_loader = data.DataLoader(
        dataset,
        batch_size=8,
        sampler=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

    model = DPC_RNN(
        sample_size=128, 
        num_seq=8, 
        seq_len=5, 
        network="resnet18", 
        pred_step=3,
    )
    model = model.to(cuda)
    checkpoint = torch.load("../ucf101-rgb-128_resnet18_dpc.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    # TODO: Complete hack? Probably doesn't work if running on multiple
    # gpus.
    hook_result = {}    
    def hook(_model, _input, output):
        embedding = output
        embedding = F.avg_pool2d(output, 4)
        embedding = embedding.squeeze(-1).squeeze(-1)
        hook_result["embedding"] = embedding.detach()
    model.observed_hidden.register_forward_hook(hook)

    # TODO: Maybe memorymap inputs and embeddings here?

    cache_path = "/tmp/dpc-cache.npz"
    if os.path.exists(cache_path):
        archive = np.load(cache_path)
        inputs = archive["arr_0"]
        embeddings = archive["arr_1"]
    else:
        inputs = []
        embeddings = []
        model.eval()
        with torch.no_grad():
            for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
                input_seq = input_seq.to(cuda)
                model(input_seq)
                inputs.extend(input_seq.cpu().detach().numpy())
                embeddings.extend(hook_result["embedding"].cpu().detach().numpy())
        
        inputs = np.array(inputs)
        inputs = np.transpose(inputs, (0, 1, 3, 4, 5, 2))
        inputs = np.reshape(inputs, (-1, 40, 128, 128, 3))
        embeddings = np.array(embeddings)
        np.savez(cache_path, inputs, embeddings)

    for _ in range(10):
        query_idx = random.randint(0, 100)
        query_input = inputs[query_idx]
        query_embedding = embeddings[query_idx]
        inputs = np.delete(inputs, query_idx, axis=0)
        embeddings = np.delete(embeddings, query_idx, axis=0)

        # TODO: Show top 3.
        numerator = embeddings @ query_embedding[:, None]
        numerator = numerator.squeeze()
        a = np.sqrt((query_embedding ** 2).sum())
        b = np.sqrt((embeddings ** 2).sum(axis=-1))
        denominator = a * b
        cosine_sim = numerator / denominator
        idx = cosine_sim.argmax()

        # TODO: Should be cosine distance.
        #st.text("finding closest neighbor")
        #sq_dist = ((query_embedding - embeddings) ** 2).sum(axis=-1)
        #idx = sq_dist.argmin()

        neighbor = inputs[idx]

        def denormalize(x):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            x = x * std + mean
            return x

        st.header("Query")
        for i in range(3):
            st.image(denormalize(query_input[i]), clamp=True)

        st.header("Closest neighbor")
        for i in range(3):
            st.image(denormalize(neighbor[i]), clamp=True)



if __name__ == "__main__":
    main()
