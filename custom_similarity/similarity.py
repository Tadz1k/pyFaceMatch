import argparse
import os
import cv2
import numpy as np
from .embeddings import get_single_embedding_from_file
from .embeddings import get_single_embedding_from_memory
from PIL import Image

def get_similarity_from_files(img1, img2, input_size=[112, 112]):
    _, embeddings1 = get_single_embedding_from_file(
        img_path=f"{img1}",
        model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
        input_size=input_size,
    )
    _, embeddings2 = get_single_embedding_from_file(
        img_path=f"{img2}",
        model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
        input_size=input_size,
    )

    # calculate similarity
    sim = np.dot(embeddings1, embeddings2.T).squeeze()
    return sim

#PIL image format is preffered
def get_similarity_from_memory(img1, img2, input_size=[112, 112]):
    embeddings1 = get_single_embedding_from_memory(
        img=img1,
        model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
        input_size=input_size,
    )
    embeddings2 = get_single_embedding_from_memory(
        img=img2,
        model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
        input_size=input_size,
    )

    # calculate similarity
    sim = np.dot(embeddings1, embeddings2.T).squeeze()
    return sim

#hybrid function
def get_similarity(img1, img2, input_size=[112, 112]):
    emb1 = None
    emb2 = None
    if type(img1) is str:
        emb1 = get_single_embedding_from_file(
            img_path=f"{img1}",
            model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
            input_size=input_size,
        )
    else:
        emb1 = get_single_embedding_from_memory(
            img=img1,
            model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
            input_size=input_size,
        )
    if type(img2) is str:
        emb2 = get_single_embedding_from_file(
            img_path=f"{img2}",
            model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
            input_size=input_size,
        )
    else:
        emb2 = get_single_embedding_from_memory(
            img=img2,
            model_root="custom_similarity\\models\\backbone_ir50_ms1m_epoch120.pth",
            input_size=input_size,
        )
    print(emb1)
    sim = np.dot(emb1, emb2[1].T).squeeze()
    return sim
