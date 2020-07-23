# from extract_face import img
from torch import zeros
from config import get_config
from glob import glob
from utils import get_pred, get_pairs_intra_label, get_pairs_inter_label, findDistance_cos, findDistance, calc_diff_and_similar
from Learner import face_learner
import numpy as np
# print(findDistance(np.array([2, 2]), np.array([1, 1])))

conf = get_config(False)
list_label_folder = glob(f"{conf.infer_dataset}/*")

list_label_folder = [folder for folder in list_label_folder if len(glob(f"{folder}/*")) != 0]


all_intra_pairs, all_inter_pairs, y_pred = [], [], np.array([])

for path in list_label_folder:
    intra_pairs = get_pairs_intra_label(path)
    all_intra_pairs += intra_pairs
    all_inter_pairs += get_pairs_inter_label(path, list_label_folder, intra_pairs)

learner = face_learner(conf, True)
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')


diff1, similar1 = calc_diff_and_similar(all_inter_pairs, all_intra_pairs, conf, learner, findDistance)
diff2, similar2 = calc_diff_and_similar(all_inter_pairs, all_intra_pairs, conf, learner, findDistance_cos)


# import seaborn as sns

# sns.set(color_codes=True)
# sns.distplot(similar1, hist=False)
# sns.distplot(diff1, hist=False)

range1 = [1, 1.2, 1.4]
range2 = [0.4, 0.7, 1]


step = 0.1
import numpy as np
y_true = np.concatenate((np.zeros(len(diff1),dtype=bool) , np.ones(len(similar1), dtype=bool)), axis=0)

y_pred = get_pred(diff1, similar1, 1.2)

# print(y_pred)

