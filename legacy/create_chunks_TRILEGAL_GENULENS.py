
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from astropy import constants as const
from astropy import units as u
import math
from astropy.constants import c, L_sun, sigma_sb, M_jup, M_earth
import sys,os

# data_tril = pd.read_csv(file_path_TRILEGAL, sep="\s+", decimal ='.', header = [0])
# data_tril['W149'] = data_tril['W149']+1.2258 #transform Vega magnitudes into AB
# data_tril = data_tril[data_tril['i']<28]

file_path_genulens = '/home/anibal-pc/large_files_roman_rubin/genulens_in_8000_pc.csv'
file_path_TRILEGAL = '/home/anibal-pc/large_files_roman_rubin/output445007434654_AB.csv'
# output_dir = '/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS'  
output_dir = '/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS/'  
# Directory to store chunks
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

header_koshimoto = ["wtj", "M_L", "D_L", "D_S", "tE", "thetaE", 
                    "piE", "piEN", "piEE", "mu_rel", "muSl", "muSb", 
                    "i_L", "iS", "iL", "fREM"]

# Step 1: Get column names for TRILEGAL
with open(file_path_TRILEGAL, 'r') as f:
    for line in f:
        if line.startswith('#'):
            columns_TRILEGAL = line[1:].strip().split()
            break

# Step 2: Read in chunks of 10,000 rows
chunk_size = 10000
max_chunks = 20  # Set the number of chunks to process

# Processing TRILEGAL
trilegal_chunks = pd.read_csv(
    file_path_TRILEGAL,
    sep=',',
    # delim_whitespace=True,
    # comment=',',
    chunksize=chunk_size,
    header=[0],
    # names=columns_TRILEGAL
)

# display
# trilegal_chunks = trilegal_chunks[trilegal_chunks['i']<28]
# trilegal_chunks['W149'] = trilegal_chunks['W149']+1.2258 #transform Vega magnitudes into AB


# Processing Genulens
genulens_chunks = pd.read_csv(
    file_path_genulens,
    chunksize=chunk_size,
    header=1,
    names=header_koshimoto
)
# Loop through chunks and save
for i, (chunk_TRILEGAL, chunk_koshimoto) in enumerate(zip(trilegal_chunks, genulens_chunks)):
    if i >= max_chunks:
        break  # Stop if max chunks are processed

    print(f"Processing and saving chunk {i + 1}")

    # Example calculation
    # chunk_koshimoto['piE'] = np.sqrt(chunk_koshimoto['piEE']**2 + chunk_koshimoto['piEN']**2)
    
    # Save each chunk to CSV
    trilegal_chunk_path = os.path.join(output_dir, f"TRILEGAL_chunk_{i + 1}.csv")
    genulens_chunk_path = os.path.join(output_dir, f"Genulens_chunk_{i + 1}.csv")
    # display(chunk_TRILEGAL.head(2))
    # display(chunk_koshimoto.head(2))
    chunk_TRILEGAL.to_csv(trilegal_chunk_path, index=False)
    chunk_koshimoto.to_csv(genulens_chunk_path, index=False)
    
    print(f"Saved TRILEGAL_chunk_{i + 1}.csv and Genulens_chunk_{i + 1}.csv")

print("Chunking and saving complete!")