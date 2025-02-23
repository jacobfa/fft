#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm
import tarfile
import shutil
import subprocess
import urllib.request

def download_file(url, dest):
    """Download a file from a URL and save it locally with a progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        desc = os.path.basename(dest)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(f"Finished downloading {dest}")

def download_all_sequential():
    """Download all files one at a time."""
    files = [
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz", "ILSVRC2012_devkit_t12.tar.gz"),
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar", "ILSVRC2012_img_val.tar"),
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar", "ILSVRC2012_img_train.tar")
    ]
    for url, dest in files:
        download_file(url, dest)

def extract_tar(tar_path, extract_to=None):
    """Extract a tar file to a specified directory."""
    if extract_to is None:
        extract_to = os.path.splitext(tar_path)[0]
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_to)
    print(f"Finished extracting {tar_path}")

def extract_training_data():
    """Extract the training data and nested archives."""
    os.makedirs("train", exist_ok=True)
    shutil.move("ILSVRC2012_img_train.tar", "train/")
    os.chdir("train")
    
    # Extract the main training tar file.
    extract_tar("ILSVRC2012_img_train.tar")
    os.remove("ILSVRC2012_img_train.tar")
    
    # For each nested tar file, create a subdirectory, extract, then remove the tar.
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".tar"):
                tar_path = os.path.join(root, file)
                subfolder = os.path.splitext(tar_path)[0]
                os.makedirs(subfolder, exist_ok=True)
                extract_tar(tar_path, subfolder)
                os.remove(tar_path)
    os.chdir("..")

def extract_validation_data():
    """Extract the validation data and prepare subfolders using the valprep script."""
    os.makedirs("val", exist_ok=True)
    shutil.move("ILSVRC2012_img_val.tar", "val/")
    os.chdir("val")
    
    # Extract the validation tar file.
    extract_tar("ILSVRC2012_img_val.tar")
    
    # Download and run the valprep.sh script to reorganize images into subfolders.
    valprep_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    valprep_script = "valprep.sh"
    print("Downloading valprep.sh...")
    urllib.request.urlretrieve(valprep_url, valprep_script)
    print("Running valprep.sh...")
    subprocess.run(["bash", valprep_script], check=True)
    os.chdir("..")

if __name__ == '__main__':
    # Step 1: Download files sequentially with progress bars.
    download_all_sequential()
    
    # Step 2: Extract training data.
    extract_training_data()
    
    # Step 3: Extract validation data and prepare subfolders.
    extract_validation_data()
