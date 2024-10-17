import numpy as np
import os
import os.path as osp
import shutil
import json
import pandas as pd
from tqdm import tqdm

################################
# reformat metacsv to metajson #
################################

if __name__ == '__main__':
    folder = '/weka/datasets/midjourney/from_kaggle'
    sample_dir = '/weka/datasets/midjourney/from_kaggle/sample/'
    os.makedirs(sample_dir, exist_ok=True)
    im_number = sorted([i for i in os.listdir(osp.join(folder, 'image/part5/')) if i.endswith('.jpg')])
    json_number = sorted([i for i in os.listdir(osp.join(folder, 'json/part5/')) if i.endswith('.json')])

    for imi, jsoni in zip(im_number[0:1000], json_number[0:1000]):
        imi_from = osp.join(folder, 'image/part5/', imi)
        jsoni_from = osp.join(folder, 'json/part5/', jsoni)
        shutil.copy(imi_from, sample_dir)
        shutil.copy(jsoni_from, sample_dir)

    debug = 1