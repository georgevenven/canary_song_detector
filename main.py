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
        
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Song Detection')
    
    # Add arguments
    parser.add_argument('folders_dir', type=str)
    parser.add_argument('weights_dir', type=str)

    # Parse the arguments
    args = parser.parse_args()
    
    # Use the arguments
    song_detector(args.folders_dir, args.weights_dir, model)