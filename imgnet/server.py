#!/usr/bin/env python3
import os
import math
import requests
import tarfile
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

########################
# Segmented Download
########################

def download_segment(url, start, end, index):
    headers = {"Range": f"bytes={start}-{end}"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    segment_filename = f"segment_{index}.part"
    total = end - start + 1
    with open(segment_filename, "wb") as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=f"Segment {index}"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return segment_filename

def merge_segments(segment_files, output_file):
    with open(output_file, "wb") as outfile:
        for seg in segment_files:
            with open(seg, "rb") as infile:
                outfile.write(infile.read())
    # Remove temporary segment files.
    for seg in segment_files:
        os.remove(seg)

def download_file_concurrently(url, output_file, num_segments=4):
    # Get the file size from headers.
    head = requests.head(url)
    if "Content-Length" not in head.headers:
        raise Exception("Failed to retrieve Content-Length. Cannot perform segmented download.")
    file_size = int(head.headers["Content-Length"])
    segment_size = math.ceil(file_size / num_segments)
    
    segments = []
    with ThreadPoolExecutor(max_workers=num_segments) as executor:
        futures = []
        for i in range(num_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size - 1, file_size - 1)
            futures.append(executor.submit(download_segment, url, start, end, i))
        for future in as_completed(futures):
            segments.append(future.result())
    
    merge_segments(segments, output_file)
    print(f"Finished downloading {output_file}")

########################
# Extraction Functions
########################

def extract_tar(tar_path, extract_to=None):
    if extract_to is None:
        extract_to = os.path.splitext(tar_path)[0]
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_to)
    print(f"Finished extracting {tar_path}")

def extract_devkit():
    """Extract the devkit into a 'devkit' folder."""
    os.makedirs("devkit", exist_ok=True)
    shutil.move("ILSVRC2012_devkit_t12.tar.gz", "devkit/")
    devkit_dir = os.path.abspath("devkit")
    os.chdir(devkit_dir)
    extract_tar("ILSVRC2012_devkit_t12.tar.gz")
    os.remove("ILSVRC2012_devkit_t12.tar.gz")
    os.chdir("..")

def extract_training_data():
    """Extract the training tar file into a 'train' folder and then extract all nested tars concurrently using 4 CPUs."""
    os.makedirs("train", exist_ok=True)
    shutil.move("ILSVRC2012_img_train.tar", "train/")
    train_dir = os.path.abspath("train")
    os.chdir(train_dir)
    
    # Extract the main training tar file.
    extract_tar("ILSVRC2012_img_train.tar")
    os.remove("ILSVRC2012_img_train.tar")
    
    # Find all nested tar files.
    tar_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".tar"):
                tar_files.append(os.path.join(root, file))
    
    # Use 4 processes to extract nested tar files concurrently.
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_tar = {}
        for tar_path in tar_files:
            subfolder = os.path.splitext(tar_path)[0]
            os.makedirs(subfolder, exist_ok=True)
            future = executor.submit(extract_tar, tar_path, subfolder)
            future_to_tar[future] = tar_path
        for future in as_completed(future_to_tar):
            tar_path = future_to_tar[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Extraction of {tar_path} generated an exception: {exc}")
    
    # Remove the now-extracted tar files.
    for tar_path in tar_files:
        if os.path.exists(tar_path):
            os.remove(tar_path)
    os.chdir("..")

def extract_validation_data():
    """Extract the validation tar file into a 'val' folder and run the valprep.sh script to reorganize images."""
    os.makedirs("val", exist_ok=True)
    shutil.move("ILSVRC2012_img_val.tar", "val/")
    os.chdir("val")
    
    # Extract the validation tar file.
    extract_tar("ILSVRC2012_img_val.tar")
    
    # Download and run the valprep.sh script.
    valprep_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    valprep_script = "valprep.sh"
    print("Downloading valprep.sh...")
    urllib.request.urlretrieve(valprep_url, valprep_script)
    print("Running valprep.sh...")
    subprocess.run(["bash", valprep_script], check=True)
    os.chdir("..")

########################
# Main Execution
########################

if __name__ == '__main__':
    # URLs and output filenames.
    files = [
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz", "ILSVRC2012_devkit_t12.tar.gz"),
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar", "ILSVRC2012_img_val.tar"),
        ("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar", "ILSVRC2012_img_train.tar")
    ]
    
    # Download each file using segmented downloads.
    for url, output_file in files:
        download_file_concurrently(url, output_file, num_segments=4)
    
    # Extract each downloaded file into its proper folder.
    extract_devkit()
    extract_training_data()
    extract_validation_data()
