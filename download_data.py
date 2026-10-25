#!/usr/bin/env python3
"""
download_data.py  —  Fetch the MedMNIST datasets used in this project and
save them as the .npy files the search scripts expect:

    <dataset>/train_images.npy   <dataset>/train_labels.npy
    <dataset>/val_images.npy     <dataset>/val_labels.npy
    <dataset>/test_images.npy    <dataset>/test_labels.npy

All images are the standard 28 x 28 MedMNIST versions.

Usage (from the repository root):
    python download_data.py                      # all four datasets
    python download_data.py breastmnist          # just one
    python download_data.py breastmnist dermamnist
"""
import argparse
import os

import numpy as np
import medmnist
from medmnist import INFO

DATASETS = ["breastmnist", "dermamnist", "retinamnist", "bloodmnist"]
SPLITS   = ["train", "val", "test"]
CACHE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".medmnist_cache")


def download(flag: str) -> None:
    data_class = getattr(medmnist, INFO[flag]["python_class"])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), flag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    print(f"{flag}:")
    for split in SPLITS:
        ds = data_class(split=split, download=True, root=CACHE, size=28)
        np.save(os.path.join(out_dir, f"{split}_images.npy"), ds.imgs)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), ds.labels)
        print(f"  {split:5s}  images {ds.imgs.shape}  labels {ds.labels.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"any of {', '.join(DATASETS)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for flag in (args.datasets or DATASETS):
        download(flag)
    print("Done.")
