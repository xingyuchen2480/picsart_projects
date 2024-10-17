import os
import hashlib
import pandas as pd
from tqdm import tqdm
import json

json_dir='/weka/datasets/midjourney/from_kaggle/json/part10/'

meta_file = '/weka/datasets/midjourney/from_kaggle/meta/prompts_part10.csv'
meta_file_new = '/weka/datasets/midjourney/from_kaggle/meta/prompts_part10_errorfree.parquet'
minfo = pd.read_csv(meta_file)
minfo_filenamenull = minfo[minfo['img_name'].isnull()]
minfo_filenamenotnull = minfo[minfo['img_name'].notnull()]
minfo_filenamenotnull = minfo_filenamenotnull.reset_index(drop=True)
del minfo_filenamenotnull['Unnamed: 0']
minfo_filenamenotnull['master_id'] = minfo_filenamenotnull['image_url'].apply(
    lambda x: x.split('/')[-1].split('.')[0])

minfonew = minfo_filenamenotnull[['master_id', 'img_name', 'prompt', 'image_url', 'timestamp']]
minfonew.to_parquet(meta_file_new)

debug=1
# Path to the directory with images
meta_store_file = meta_file_new
hash_store_file = '/weka/datasets/midjourney/from_kaggle/meta/hash_part10.parquet'

df1=minfonew
df1['img_name'] = df1['img_name'].str.replace('.png', '.jpg')
df2=pd.read_parquet(hash_store_file)
#df2 = df2.rename(columns={'ID': 'img_name'})
# Save the merged DataFrame
merged_df = pd.merge(df1, df2, on='img_name', how='inner')
#merged_df.to_parquet('/weka/datasets/midjourney/from_kaggle/meta/prompts_part2_errorfree_with_hash.parquet', index=False)


def write_row_to_json(row):
    # Convert the row to a dictionary
    row_dict = row.to_dict()
    # Use img_name as the JSON file name, replacing .jpg or .png with .json
    json_file_name = row['img_name'].replace('.jpg', '.json').replace('.png', '.json')
    json_file_path = os.path.join(json_dir, json_file_name)
    # Write the dictionary to a JSON file with indentation for readability
    with open(json_file_path, 'w') as json_file:
        json.dump(row_dict, json_file, indent=4)
os.makedirs(json_dir, exist_ok=True)
# Iterate through each row in the DataFrame and write it to a JSON file with tqdm progress bar
for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Writing JSON files", unit="file"):
    write_row_to_json(row)

print("JSON files created for each row.")

debug=1