
from pathlib import Path
from config import get_config
from data.data_pipe import load_bin, load_mx_rec_custom
import argparse

if __name__ == '__main__':
    conf = get_config()
    rec_path = Path("/media/2tb/data/DeepGlint/faces_glint")
    
    bin_files = ['agedb_30', 'cfp_fp', 'lfw',
                 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']

    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i]+'.bin'),
                 rec_path/bin_files[i], conf.test_transform)