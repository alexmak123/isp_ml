### all these scripts were used during developing a solution for proper data format

cappa_koef.py - script, that was used to count cohen's kappa score between two experts

cutting_slide_in_tiles.py - script, that was used to cut histological slides WSI into tiles of the given size (+ check if the image is mostly background)

dataset_to_yaml.py - script, that was used to convert labeled dataset from CVAT to splited on train/val/test dataset in yaml-format 

xml_parser.py - script, that was used to parse xml with annotations, received from CVAT to folders with "images" and "labels" with .txt with keypoints of labeled objects on this images

generate_extended_dataset_images.py - script, that was used to add context (bounding boxes) to dataset images for proper cells annotation  on the borders while labelling in CVAT

coord_transform.py - script, that was used to transform coordinates in labels .txt to proper format
