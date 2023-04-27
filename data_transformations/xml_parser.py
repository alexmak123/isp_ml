#!/usr/bin/env python
# coding: utf-8


import xml.etree.ElementTree as ET
import os
import argparse


def empty_folder(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

            
def parsing(annotation_name, classes, save, preserve=False):
    if not os.path.exists(annotation_name):
        raise BaseException("No such file:" + annotation_name)
    print(f'annotation_name >{annotation_name}<')
    tree = ET.parse(annotation_name)
    root = tree.getroot()

    if not os.path.isdir(save):
        print("Creating directory " + save + "...")
        os.mkdir(save)
    else:
        if not preserve:
            print('Destination folder is not empty. Removing all files...')
            empty_folder(save)

    for child_of_root in root[2:]:
        print(child_of_root)
        name = os.path.splitext(child_of_root.get('name'))[0]
        with open(str(os.path.join(save, name) + '.txt'), 'w') as label_file:
            for i, image in enumerate(child_of_root):
                label = classes[image.get('label')]
                points = image.get('points').split(';')
                for j, point in enumerate(points):
                    point_list = point.split(',')
                    point_list.append(str(label))
                    label_file.write(' '.join(point_list))
                    n = len(child_of_root)
                    k = len(points)
                    if i != (len(child_of_root) - 1) or j != (len(points) - 1):
                        label_file.write('\n')                    
                        
                        
CLASSES = {'stroma': 0, 'epithelium': 1, 'other': 2}
path_to_xml = "/home/alexmak123/test_cappa_koef/endometrium_with_plasmatic/task_mytask_valid/annotations.xml"
save = "/home/alexmak123/test_cappa_koef/endometrium_with_plasmatic/task_mytask_valid"
preserve=True
parsing(path_to_xml, CLASSES, save, preserve)

