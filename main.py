import os 
from utils import mat_to_npy
from tqdm import tqdm

src = "/home/george-vengrovski/Documents/lab_canary_data"
dst = "/home/george-vengrovski/Documents/canary_song_detector/data"
format = ".mat"

files = [""]

src_files = os.listdir(src)
    
if format == ".mat":
    for bird in src_files:
        for day in os.listdir(os.path.join(src, bird)): 
            d = os.path.join(src, bird, day)
            for file in tqdm(os.listdir(d), desc=f"Processing files for bird {bird} on day {day}"):
                npy_file = mat_to_npy(src = os.path.join(src, bird, day, file), dst= dst, filename= file)
        
                
