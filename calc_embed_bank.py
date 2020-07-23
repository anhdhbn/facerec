# from extract_face import img
from config import get_config
from glob import glob
from utils import calc_embed_bank
from Learner import face_learner
import numpy as np
import pickle
# print(findDistance(np.array([2, 2]), np.array([1, 1])))

conf = get_config(False)
images_path = glob(f"{conf.facebank_dataset}/*/*")[:]

learner = face_learner(conf, True)
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')


embeddings, labels = calc_embed_bank(images_path, learner, conf)

banks = {}
for label in set(labels):
    banks[label] = []

for embed, label in zip(embeddings, labels):
    banks[label].append(embed)

results = [(np.mean(banks[label], axis=0), label) for label in banks]

with open(conf.embedding_path, "wb") as file_embedding:
    pickle.dump(results, file_embedding)
