from math import factorial
import cv2
import argparse
from pathlib import Path
from PIL import Image
import pickle
from config import get_config
from PIL import Image
from preprocess_ailab_img import create_intra_pairs

if __name__ == "__main__":
    conf = get_config()

    save_path = Path('/media/2tb/data/detected_img')
    intra_pair_path = save_path/"intra_pair_IPCGANs_10k.pkl"

    all_intra_pairs = create_intra_pairs("/home/linhnv/projects/IPCGANs-Pytorch/aged_imgs", 10000)
    print(len(all_intra_pairs))
    with open(intra_pair_path, 'wb') as file_val_images:
        pickle.dump(all_intra_pairs, file_val_images)
