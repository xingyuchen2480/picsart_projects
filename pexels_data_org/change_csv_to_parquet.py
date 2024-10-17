import pandas as pd

# Path to the CSV file
csv_file = 'cleaned_images.csv'

# Path to save the Parquet file
parquet_file = 'clean_images.parquet'


df = pd.read_csv(csv_file)
df.to_parquet(parquet_file, engine='pyarrow')

print(f"CSV file {csv_file} successfully converted to Parquet file {parquet_file}")
