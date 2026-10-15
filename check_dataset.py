from pathlib import Path 
import numpy as np 

data_dir = Path("v1_data")

for folder in sorted(data_dir.iterdir()):
    if folder.is_dir():
        files = list(folder.glob("*.npy"))
        print(f"{folder.name}: {len(files)} recordings")

        if files:
            sample = np.load(files[0])
            print(f" Shape: {sample.shape}")