#!/usr/bin/env python
# coding: utf-8


import os
import random
import shutil


path_to_dataset = "/home/alexmak123/result_plasmatic_dataset_373/dataset"
labels_dir_dataset = os.path.join(path_to_dataset, 'labels')
images_dir_dataset = os.path.join(path_to_dataset, 'images')
train_lists_dir = os.path.join('/home/alexmak123/result_plasmatic_dataset_373/trainval/train/lists')
val_lists_dir = os.path.join('/home/alexmak123/result_plasmatic_dataset_373/trainval/valid/lists')


path_to_test = "/home/alexmak123/result_plasmatic_dataset_373/test"
labels_dir_test = os.path.join(path_to_test, 'labels')
images_dir_test = os.path.join(path_to_test, 'images')
test_lists_dir = os.path.join('/home/alexmak123/result_plasmatic_dataset_373/test/lists')
test_exists = False


train_size = 0.7
val_size = 0.3
if test_exists:
    val_size = 0.2
    test_size = 0.1


if not test_exists:
    # Удаляем все файлы в labels_dir_test и images_dir_test
    for file in os.listdir(labels_dir_test):
        os.remove(os.path.join(labels_dir_test, file))
    for file in os.listdir(images_dir_test):
        os.remove(os.path.join(images_dir_test, file))
    for file in os.listdir(test_lists_dir):
        os.remove(os.path.join(test_lists_dir, file))
        
        
def move_txt_files():
    txt_files = [f for f in os.listdir(labels_dir_dataset) if f.endswith('.txt')]
    random.shuffle(txt_files)
    
    if test_exists:
        num_train_files = int(len(txt_files) * train_size)
        num_val_files = int(len(txt_files) * val_size)
        num_test_files = int(len(txt_files) * test_size)
        
        train_files = txt_files[:num_train_files]
        val_files = txt_files[num_train_files:num_train_files+num_val_files]
        test_files = txt_files[num_train_files+num_val_files:]
        
        for files, target_dir in [(train_files, train_lists_dir), (val_files, val_lists_dir), (test_files, test_lists_dir)]:
            with open(os.path.join(target_dir, 'images.txt'), 'w') as img_f, \
                open(os.path.join(target_dir, 'labels.txt'), 'w') as lbl_f:
                for file in files:
                    img_file = file.replace('.txt', '.png')
                    img_f.write(f'../../../dataset/images/{img_file}\n')
                    lbl_f.write(f'../../../dataset/labels/{file}\n')
                
        for file in test_files:
            src_label_path = os.path.join(labels_dir_dataset, file)
            src_image_path = os.path.join(images_dir_dataset, file.replace('.txt', '.png'))
            dst_label_path = os.path.join(labels_dir_test, file)
            dst_image_path = os.path.join(images_dir_test, file.replace('.txt', '.png'))
            shutil.copy(src_label_path, dst_label_path)
            shutil.copy(src_image_path, dst_image_path)
    
    else:
        num_train_files = int(len(txt_files) * train_size)
        num_val_files = len(txt_files) - num_train_files
        
        train_files = txt_files[:num_train_files]
        val_files = txt_files[num_train_files:]
        
        for files, target_dir in [(train_files, train_lists_dir), (val_files, val_lists_dir)]:
            with open(os.path.join(target_dir, 'images.txt'), 'w') as img_f, \
                open(os.path.join(target_dir, 'labels.txt'), 'w') as lbl_f:
                for file in files:
                    img_file = file.replace('.txt', '.png')
                    img_f.write(f'../../../dataset/images/{img_file}\n')
                    lbl_f.write(f'../../../dataset/labels/{file}\n')


if __name__ == "__main__":
    move_txt_files()




