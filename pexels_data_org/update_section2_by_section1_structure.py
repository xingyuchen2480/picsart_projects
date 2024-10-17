import pandas as pd
df_sec1=pd.read_parquet('/weka/datasets/pexels_section2/section1/meta/pexels_section1.parquet')
df_sec2=pd.read_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_metadata_section2.parquet')
df_sec2['HW'] = df_sec2.apply(lambda row: [row['WIDTH'], row['HEIGHT']], axis=1)

# Rename columns to match the structure of df_sec1
df_sec2 = df_sec2.rename(columns={
    'LOCAL_PATH': 'OLD_PATH',
    'HASH': 'HASH',   # HASH remains the same
    'TITLE': 'TITLE', # TITLE remains the same
    'URL': 'URL',     # URL remains the same
    'ID': 'ID'        # ID remains the same
})

# Add the 'DUPLICATE_OF' column, filling with None for now
df_sec2['DUPLICATE_OF'] = None

# Drop 'WIDTH' and 'HEIGHT' since we now have 'HW'
df_sec2 = df_sec2.drop(columns=['WIDTH', 'HEIGHT'])

# Reorder columns to match df_sec1
df_sec2 = df_sec2[['ID', 'OLD_PATH', 'URL', 'HASH', 'HW', 'TITLE', 'DUPLICATE_OF']]

# Check the first row to confirm the structure is correct
print(df_sec2.iloc[0])

# merged_df = pd.concat([df_sec1, df_sec2], ignore_index=True)
# df_unique = merged_df.drop_duplicates(subset='HASH', keep=False)
# df_unique['HW'] = df_unique['HW'].apply(tuple)
# df_sec1['HW'] = df_sec1['HW'].apply(tuple)

# diff_df = pd.concat([df_unique, df_sec1]).drop_duplicates(keep=False)
df_sec2.to_parquet('/weka/datasets/pexels_section2/section2/meta/pexels_section_noterrfree.parquet')
debug=1