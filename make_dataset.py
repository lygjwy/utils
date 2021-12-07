#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def walk_dir(dir_path):
    root_path = dir_path.parents[1]
    image_paths = []
    targets = []
    classes = []
    
    for sub_item_path in dir_path.iterdir():
        if sub_item_path.is_dir():
            # labeled dataset
            classes.append(str(sub_item_path.relative_to(dir_path)))
        else:
            # unlabeled dataset
            image_name = sub_item_path.name
            if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
                image_paths.append(str(dir_path.relative_to(root_path) / image_name))


    classes = sorted(classes)
    class_to_idx = {category: i for i, category in enumerate(classes)}

    if len(classes) > 0:
        # labeled dataset
        for category in sorted(classes):
            class_path = dir_path / category
            for class_sub_item_path in class_path.iterdir():
                class_sub_name = class_sub_item_path.name
                if class_sub_item_path.is_file() and has_file_allowed_extension(class_sub_name, IMG_EXTENSIONS):
                        image_paths.append(str(class_path.relative_to(root_path) / class_sub_name))
                        targets.append(class_to_idx[category])
    return image_paths, targets, classes


def save_labeled_entry(image_paths, targets, file_path):
    lines = [' '.join([image_path, str(target)]) for image_path, target in zip(image_paths, targets)]
    content = '\n'.join(lines)
    with open(file_path, 'w') as f:
        f.write(content)


def save_unlabeled_entry(image_paths, file_path):
    content = '\n'.join(image_paths)
    with open(file_path, 'w') as f:
        f.write(content)


def make_dataset(dataset_path):
    # dataset_path E:\数据集\Dataset, absolute path
    train_path = Path(dataset_path) / 'train'
    test_path = Path(dataset_path) / 'test'
    
    train_image_paths, train_targets, train_classes = [], [], []
    if train_path.is_dir():
        train_image_paths, train_targets, train_classes = walk_dir(train_path)
    
    test_image_paths, test_targets, test_classes = [], [], []
    if test_path.is_dir():
        test_image_paths, test_targets, test_classes = walk_dir(test_path)
    
    # save to file
    train_entry_path = Path(dataset_path) / 'train.txt'
    if len(train_image_paths) > 0:
        if len(train_targets) > 0:
            save_labeled_entry(train_image_paths, train_targets, train_entry_path)
        else:
            save_unlabeled_entry(train_image_paths, train_entry_path)
    
    test_entry_path = Path(dataset_path) / 'test.txt'
    if len(test_image_paths) > 0:
        if len(test_targets) > 0:
            save_labeled_entry(test_image_paths, test_targets, test_entry_path)
        else:
            save_unlabeled_entry(test_image_paths, test_entry_path)
    
    # save classes.txt
    # if len(train_classes) != len(test_classes):
    #     raise RuntimeError('---> train classes not equal to test classes')
    classes_txt = Path(dataset_path) / 'classes.txt'
    if len(train_classes) > 0:
        # labeled dataset
        with open(classes_txt, 'w') as cf:
            cf.write(str(train_classes))
        return

    if len(test_classes) > 0:
        # labeled dataset
        with open(classes_txt, 'w') as cf:
            cf.write(str(test_classes))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make dataset')
    parser.add_argument(
        '--root',
        help='path to root dir which stores dataset dirs',
        default='E:/数据集'
    )
    parser.add_argument(
        '--datasets',
        help='dataset list',
        nargs='*',
        required=True
    )
    args = parser.parse_args()

    for ds in args.datasets:
        make_dataset(os.path.join(args.root, ds))