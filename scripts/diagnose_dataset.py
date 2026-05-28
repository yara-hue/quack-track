#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import _list_sequences, _read_annotation


def diagnose(data_root):
    print(f'Data root: {data_root}')
    print(f'Exists: {os.path.isdir(data_root)}')
    if not os.path.isdir(data_root):
        print('ROOT NOT FOUND')
        return

    seq_dir = os.path.join(data_root, 'data_seq')
    anno_dir = os.path.join(data_root, 'anno')
    print(f'data_seq exists: {os.path.isdir(seq_dir)}')
    print(f'anno exists: {os.path.isdir(anno_dir)}')

    seq_names = _list_sequences(data_root)
    print(f'\n_list_sequences found: {len(seq_names)} seqs')
    if seq_names:
        print(f'First 5: {seq_names[:5]}')

    found = 0
    for name in seq_names:
        anno_path = os.path.join(anno_dir, f'{name}.txt')
        if os.path.isfile(anno_path):
            bboxes = _read_annotation(anno_path)
            valid = sum(1 for b in bboxes if b is not None)
            total = len(bboxes)
            found += 1
            if found <= 3:
                print(f'  {name}: {total} frames, {valid} valid')

    print(f'\nSeqs with matching annotations: {found}')

    # Also scan raw for comparison
    dir_entries = sorted(os.listdir(seq_dir)) if os.path.isdir(seq_dir) else []
    print(f'\nDirect entries in data_seq: {len(dir_entries)}')
    for d in dir_entries[:5]:
        img_d = os.path.join(seq_dir, d, 'img')
        has_img = os.path.isdir(img_d)
        anno_p = os.path.join(anno_dir, f'{d}.txt')
        has_anno = os.path.isfile(anno_p)
        print(f'  {d}: img={has_img}, anno={has_anno}')


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else './data/UAV123_10fps'
    diagnose(root)
