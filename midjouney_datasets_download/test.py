import os
import os.path as osp
import pandas as pd
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    download_folder = '/weka/datasets/midjourney/from_kaggle/download/part10_raw'
    meta_file = '/weka/datasets/midjourney/from_kaggle/meta/prompts_part10.csv'
    
    df=pd.read_csv(meta_file)
    image_links = df['image_url'].apply(lambda x: os.path.basename(x))  # Extract filenames from links


    downloaded_files = set(os.listdir(download_folder))
    missing_images = set(image_links) - downloaded_files

    minfo = pd.read_csv(meta_file)
    minfo_status = {
        'filename_none' : [],
        'not_downloaded': [],
        'not_4_crop': [],
    }
    minfo_temp_storage = {}


    for idx, rowi in tqdm(minfo.iterrows(), total=len(minfo)):
        mid0 = rowi['image_url'].split('/')[-1].split('.')[0]
        if rowi['img_name'] == rowi['img_name']:
            mid1 = rowi['img_name'].split('__')[0]
        else:
            minfo_status['filename_none'].append(idx)
            continue

        if mid0 != mid1:
            print(idx, "download id doesn't match")

        if not osp.isfile(osp.join(download_folder, '{}.png'.format(mid0))):
            minfo_status['not_downloaded'].append(idx)

        if mid1 in minfo_temp_storage:
            minfo_temp_storage[mid0] += 1
        else:
            minfo_temp_storage[mid0] = 1

    minfo_temp_storage_err = {ni:vi for ni, vi in minfo_temp_storage.items() if vi!=4}

    for midi, _ in minfo_temp_storage_err.items():
        minfo_status['not_4_crop'].append(midi)

    for idx in minfo_status['filename_none']:
        url = minfo.iloc[idx]['image_url']
        mid = url.split('/')[-1].split('.')[0]
        errfile = osp.join(download_folder, '{}.err'.format(mid))

        with open(errfile, 'w') as f:
            f.write(url)
    debug=1
