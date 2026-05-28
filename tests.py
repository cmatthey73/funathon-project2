# %% import

import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

import polars as pl

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

print(df.head())
print(f"Total rows: {len(df)}")

print("nb nace :", df["code"].n_unique())

# %% prepare data


