from pathlib import Path
import hashlib
from PIL import Image
import numpy as np


def file_hash(path, chunk_size=8192):
    """Return SHA256 hash of a file."""
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)

    return sha.hexdigest()

def compare_png(file1, file2):
    img1 = Image.open(file1).convert("RGBA")
    img2 = Image.open(file2).convert("RGBA")

    if img1.size != img2.size:
        return False, "Different dimensions"

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    if np.array_equal(arr1, arr2):
        return True, "Identical pixels"

    diff_pixels = np.sum(np.any(arr1 != arr2, axis=2))

    return False, f"{diff_pixels} pixels differ"


def compare_folders(folder1, folder2):
    """
    Compare files with the same relative paths in two folders.
    Returns files whose contents differ.
    """

    folder1 = Path(folder1)
    folder2 = Path(folder2)

    differences = []

    for file1 in folder1.rglob("*"):
        if not file1.is_file():
            continue

        relative = file1.relative_to(folder1)
        file2 = folder2 / relative

        if not file2.exists():
            differences.append((relative, "Missing in folder2"))
            continue

        if file1.suffix.lower() == ".png":
            identical, message = compare_png(file1, file2)
            if not identical:
                differences.append((relative, message))
            continue

        else:
            hash1 = file_hash(file1)
            hash2 = file_hash(file2)

            if hash1 != hash2:
                differences.append((relative, "Different"))

    return differences


diffs = compare_folders(
    "D:\Ananta\Programming\Python\CSE220\Offline1\CSE-220_Offline_Convolution\expected_outputs",
    "D:\Ananta\Programming\Python\CSE220\Offline1\CSE-220_Offline_Convolution\outputs"
)

for file, status in diffs:
    print(status, ":", file)