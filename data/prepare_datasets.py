#!/usr/bin/env python
import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile


COCO_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
}


def download_coco_subset(output_dir, max_images=5000):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'train2017.zip')
    extract_path = os.path.join(output_dir, 'train2017')

    if os.path.isdir(extract_path):
        print(f'COCO train2017 already exists at {extract_path}')
        return

    print('Downloading COCO train2017 (~18GB)...')
    print('This may take a while. Consider using a smaller subset.')
    urllib.request.urlretrieve(COCO_URLS['train2017'], zip_path)

    print('Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)
    print(f'COCO train2017 extracted to {extract_path}')


def link_dataset(src, dst):
    if os.path.isdir(dst) or os.path.islink(dst):
        print(f'{dst} already exists')
        return
    if os.path.isdir(src):
        os.symlink(src, dst, target_is_directory=True)
        print(f'Linked {src} -> {dst}')
    else:
        print(f'Source {src} not found. Please download UAV123/UAV20L manually.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uav123', type=str, help='Path to UAV123 dataset')
    parser.add_argument('--uav20l', type=str, help='Path to UAV20L dataset')
    parser.add_argument('--coco', action='store_true', help='Download COCO subset')
    parser.add_argument('--coco-max-images', type=int, default=5000)
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))

    if args.uav123:
        link_dataset(args.uav123, os.path.join(data_dir, 'UAV123_10fps'))
    if args.uav20l:
        link_dataset(args.uav20l, os.path.join(data_dir, 'UAV20L'))
    if args.coco:
        download_coco_subset(data_dir, max_images=args.coco_max_images)

    print('Dataset setup complete.')


if __name__ == '__main__':
    main()
