from math import factorial
import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import MAXYEAR, datetime
import glob
import os
import tqdm
import pickle
from utils import face_reader, take_random_image_in_folder, clear_non_image, get_pairs_intra_label
import math

from config import get_config

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from data_augmentation import augment_img

def clear_path(paths):
    image_type = [".png", ".jpg", ".jpeg"]
    return [path for path in paths if os.path.splitext(path)[1] in image_type]

def detect_face(data_path, faces_folder, no_face_folder):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if not faces_folder.exists():
        faces_folder.mkdir(parents=True, exist_ok=True)

    if not no_face_folder.exists():
        no_face_folder.mkdir(parents=True, exist_ok=True)

    list_files = glob.glob(f'{data_path}/*')
    list_imgs = clear_path(list_files)
    print(f"[INFO] num imgs: {len(list_imgs)}")

    num_face = 0
    num_no_face = 0
    mtcnn = MTCNN()

    samples_tqdm = tqdm.tqdm(iter(list_imgs), position=0, leave=True)
    for path_img in samples_tqdm:
        raw_img =  Image.open(path_img).convert('RGB')
        name_img = os.path.basename(path_img)
        face_img = mtcnn.align_and_take_one(raw_img)
        # print(f"Shape img: {face_img.size}")
        # face_img = face_img.resize((112, 112), Image.ANTIALIAS)
        if face_img == None:
            # print("[INFO] No face detected")
            raw_img.save(no_face_folder/f"{name_img}")
            num_no_face += 1
        else:
            face_img.save(faces_folder/f"{name_img}")
            num_face += 1

        samples_tqdm.set_description(f"Face: {num_face}, no face: {num_no_face}")

        # print(f"[INFO] Num faces: {num_face}, Num img detected face false: {num_no_face} \r")

def create_random_inter_pair(folder_path, num_pair=100):
    print(f"[INFO] Start create {num_pair} random inter pair.")
    list_inter_pair = []
    for i in tqdm.tqdm(range(num_pair)):
        
        img1 = take_random_image_in_folder(folder_path=folder_path, take=1)[0]
        img2 = take_random_image_in_folder(folder_path=folder_path, take=1)[0]

        if os.path.basename(img1) != os.path.basename(img2):
            list_inter_pair.append((img1,img2))

    return list_inter_pair

def create_sequence_inter_pair(img_folder, num_pair=100):
    print(f"[INFO] Start create {num_pair} sequence inter pair.")
    list_inter_pair = []

    list_imgs_path = clear_non_image(img_folder)
    max_pair = math.factorial(len(list_imgs_path)) / ( math.factorial(2) * math.factorial(len(list_imgs_path) - 2)) 
    max_pair = int(round(max_pair))
    print(f"[INFO] Maximum pairs can be create: {max_pair}")

    if num_pair > max_pair:
        print("[WAR] Num pairs > maximum pairs. RETURN maximum pairs.")

    for i in range(len(list_imgs_path)):
        if len(list_inter_pair) >= num_pair: break
        for j in range(i + 1, len(list_imgs_path)):
            
            list_inter_pair.append((list_imgs_path[i], list_imgs_path[j]))
            if len(list_inter_pair) >= num_pair: break

    return list_inter_pair

def create_intra_pairs(img_folder, num_pair=100):
    all_intra_pairs = []
    dir_imgs = glob.glob(f"{img_folder}/*")
    # print(len(dir_imgs))
    for path in tqdm.tqdm(dir_imgs[:num_pair]):
        intra_pairs = get_pairs_intra_label(path)
        if len(intra_pairs) >= 1:
            all_intra_pairs += intra_pairs[0]

    return all_intra_pairs

if __name__ == "__main__":
    conf = get_config()

    data_path = Path('/media/2tb/data/anhNCT12_20_26')
    save_path = Path('/media/2tb/data/detected_img')
    img_folder = save_path/"faces"
    no_face_folder = save_path/"no_faces"
    inter_pair_path = save_path/"inter_pair_10k.pkl"
    intra_pair_path = save_path/"intra_pair_10k.pkl"
    augmented_folder = "/media/2tb/data/detected_img/augmentation"

    detect_face(data_path, img_folder, no_face_folder=no_face_folder)

    list_path_imgs = glob.glob(img_folder/"*")
    for i in tqdm.tqdm(range(len(list_path_imgs))):
        augment_img(list_path_imgs[i], output=augmented_folder)

    # all_inter_pairs =  create_random_inter_pair(img_folder, num_pair=100000)
    print("[INFO] Creating inter pair.")
    all_inter_pairs = create_sequence_inter_pair(img_folder, 10000)
    print(f"[INFO] Len inter pairs: f{len(all_inter_pairs)}")
    with open(inter_pair_path, 'wb') as file_val_images:
        pickle.dump(all_inter_pairs, file_val_images)

    print("[INFO] Creating intra pair.")
    all_intra_pairs = create_intra_pairs(augmented_folder, 10000)
    print(f"[INFO] Len intra pairs: f{len(all_intra_pairs)}")
    with open(intra_pair_path, 'wb') as file_val_images:
        pickle.dump(all_intra_pairs, file_val_images)

    # intra_pair_path = save_path/"intra_pair_IPCGANs_10k.pkl"
    # all_intra_pairs = create_intra_pairs("/home/linhnv/projects/IPCGANs-Pytorch/aged_imgs", 2000)
    # print(len(all_intra_pairs))
    # with open(intra_pair_path, 'wb') as file_val_images:
    #     pickle.dump(all_intra_pairs, file_val_images)
