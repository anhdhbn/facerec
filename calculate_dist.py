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

def cal_embeds():
    learner = face_learner(conf, True)

    if conf.device.type == 'cpu':
        learner.load_state('cpu_final.pth', True, True)
    else:
        learner.load_state('ir_se50.pth', False, True)
    learner.model.eval()
    print('learner loaded')

    base_path = Path("/media/2tb/data/detected_img")
    list_imgs = glob.glob("/media/2tb/data/detected_img/faces/*")
    print(f"num imgs: {len(list_imgs)}")

    list_embs = None
    list_paths = []

    ds = RawBankDataset(list_imgs)
    loader = DataLoader(ds, batch_size=conf.batchsize_infer, shuffle=True, num_workers=8)

    i = 1
    for imgs, paths in tqdm.tqdm(iter(loader)):
        imgs = imgs.to(conf.device)
        tmp = learner.model(imgs).cpu().detach().numpy()  # (32, 512)
        if list_embs is None: list_embs = tmp
        else: list_embs = np.concatenate([list_embs, tmp], axis=0)
        # print(list_embs.shape)

        list_paths += paths
        # print(len(list_paths))
        
        i += 1
        if i >=100: break

    with open("list_embs.pkl", 'wb') as file_train_images:
        pickle.dump(list_embs, file_train_images)

    with open("list_paths.pkl", 'wb') as file_train_images:
        pickle.dump(list_paths, file_train_images)

def cal_dist():
    with open("list_embs.pkl", 'rb') as file_images:
        list_embs = pickle.load(file_images)

    with open("list_paths.pkl", 'rb') as file_images:
        list_paths = pickle.load(file_images)

    # list_embs = list_embs[:10]
    # list_paths = list_paths[:10]
    # [m*512] 
    dist = []
    embeddings_path = []
    print(f"[INFO] Num embeddings: {len(list_embs)}")
    check_usage()
    print("[INFO] Creating 2 embeddings.....")
    for index1 in range(len(list_embs)):
        for index2 in range(index1 + 1, len(list_embs)):
            dist.append(np.sum(np.square(list_embs[index1] - list_embs[index2])))
            embeddings_path.append({"path_1": list_paths[index1], "path_2": list_paths[index2]})
            print(f"[INFO] Size off var embeddings_path: {sys.getsizeof(embeddings_path) / 1024 / 1024} Mb(s) \r", end="\r")

    print("[INFO] Created 2 embedding.")
    check_usage()
    print("[INFO] Calculating diff...")

    # diff = np.subtract(embeddings1, embeddings2)
    check_usage()

    print("[INFO] Cal diff done.")

    print(len(embeddings_path))
    print(len(dist))
    # # print(dist)

    with open("dist.pkl", 'wb') as file_train_images:
        pickle.dump(dist, file_train_images)

    with open("embeddings_path.pkl", 'wb') as file_train_images:
        pickle.dump(embeddings_path, file_train_images)

if __name__ == "__main__":
    # cal_embeds()
    cal_dist()