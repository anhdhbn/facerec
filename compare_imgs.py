import glob 
from pathlib import Path
from Learner import face_learner
# from extract_face import img
from config import get_config
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils import calc_embed_bank
from PIL import Image
import numpy as np
import tqdm
import sys
import pickle
from check_usage import check_usage

from data.data_pipe import RawBankDataset
# print(findDistance(np.array([2, 2]), np.array([1, 1])))

conf = get_config(False)

def compare_imgs(threshold=1.1):
    with open("dist.pkl", 'rb') as file_images:
        dist = pickle.load(file_images)

    with open("embeddings_path.pkl", 'rb') as file_images:
        embeddings_path = pickle.load(file_images)
    dist = np.array(dist)
    # Threshold
    got_threshold_dist = np.argwhere(dist < threshold)
    got_threshold_dist = got_threshold_dist.reshape(-1)
    # print(got_threshold_dist)

    embeddings_path = np.array(embeddings_path)
    # print(embeddings_path[got_threshold_dist])
    embeddings_path = embeddings_path[got_threshold_dist]

    dir_path = Path(f"/home/linhnv/projects/facerec/issame_imgs_{threshold}")
    dir_path.mkdir(parents=True, exist_ok=True)
    for index in tqdm.tqdm(range(len(embeddings_path))):
        save_path = dir_path / f"{index}.jpg"

        images = [Image.open(embeddings_path[index]["path_1"]), Image.open(embeddings_path[index]["path_2"])]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save(save_path)

if __name__ == "__main__":
    compare_imgs(0.8)
