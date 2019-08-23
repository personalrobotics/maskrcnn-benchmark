
# coding: utf-8

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import collections
import json
from PIL import Image, ImageEnhance
import shutil

from transform import resize, random_flip, random_crop, center_crop
import random


SEG_IMG_DIR = "./segmented_images"
SEG_ANNO_DIR = SEG_IMG_DIR+"/annotations"
BLUE_RANGE = ([98,0,0], [250,255,255])

ORI_IMG_DIR = "/home/guohaz/hard_data/Data/foods/bite_selection_package/data/skewering_positions_spanet_all/cropped_images"


def crop_img(path, img, xml):
    '''
    Crop the object out of the image by using the bounding box in the label file
    @require xml is a .xml file and contain bounding box with coordinates
    @param img_path The image path
    @param xml_path The xml file path
    '''
    img = cv2.imread(os.path.join(path, 'images/'+img))
    num_boxes = 0
    tree = ET.parse(os.path.join(path, 'xmls/'+xml))
    root = tree.getroot()
    bboxes = collections.defaultdict(list)

    # find the bounding box coor
    for node in root:
        if node.tag == 'object':
            obj_name = node.find('name').text
        if node.find('bndbox') is None:
            continue
        xmin = int(node.find('bndbox').find('xmin').text)
        ymin = int(node.find('bndbox').find('ymin').text)
        xmax = int(node.find('bndbox').find('xmax').text)
        ymax = int(node.find('bndbox').find('ymax').text)
        if obj_name not in bboxes:
            bboxes[obj_name] = list()
        bboxes[obj_name].append([xmin, ymin, xmax, ymax])

        save_seg_img(img, bboxes)


def save_seg_img(img, bboxes):
    '''
    Store the image to SEG_IMG_DIR with name as 'img_name_obj_name_xmin_ymin'.png
    @param img The oringial image
    @param bboxes The bounding boxes for every objects in the image
    '''
    margin = 2
    for obj_name in sorted(bboxes):
        bbox_list = bboxes[obj_name]
        for bbox in bbox_list:
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(img.shape[1], xmax + margin)
            ymax = min(img.shape[0], ymax + margin)

            cropped_img = img[ymin:ymax, xmin:xmax]
            save_path = os.path.join(
                SEG_IMG_DIR, '{0}_{1}_{2:04d}{3:04d}.png'.format(
                    xml[:-4], obj_name, xmin, ymin))
            print(save_path)
            cv2.imwrite(save_path, cropped_img)


def extract_object(img_name):
    '''
    Extract the food item from the blue background. Store the image to SEG_IMG_DIR/obj, 
    with name as 'img_name_obj'.png
    '''
    img = cv2.imread(os.path.join(SEG_IMG_DIR, img_name+'.png'))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(BLUE_RANGE[0])
    upper_blue = np.array(BLUE_RANGE[1])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue) -255
    take = np.where(mask==0, mask, -mask)
    
    res = cv2.bitwise_and(img,img, mask=take) 
    cv2.imwrite(os.path.join(SEG_IMG_DIR+"/obj", img_name+'_obj.png'), res)
  


def detect_edge(img):
    '''
    Get the contour of the object by first finding the edge graph. 
    Return the contour coordinates with the bounding box.
    @param img The image object
    @return coor The contour coordinates in form of [x1, y1, x2, y2, ...]
    @return [x_max, y_max, x_min, y_min] The bounding box
    
    Note:
    If the image already has bounding box, we can change the code to extract the data.
    However, since we are using the cropped image for now, it is unnecessary to have a bounding box
    '''
    edges = cv2.Canny(img,100,200)
    contour,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # use edges.copy() if want to keep
    coor = []
    x_max, x_min = 0.0,float(img.shape[1])#float('inf')
    y_max, y_min = 0.0,float(img.shape[0])#float('inf')
    for part in contour:
        for xy in part:
            x, y = 1.0 * xy[0][0], 1.0 * xy[0][1]
            x_max = max(x_max, x)
            x_min = min(x_min, x)
            y_max = max(y_max, y)
            y_min = min(y_min, y)
            coor.append(x)
            coor.append(y)
    return coor, [x_max, y_max, x_min, y_min]


def write_coco_anno(label, img_name, img, coor, bnd_box, idx, category_id):
    '''
    Write and save a COCO format label.
    @param label The json directory object
    @param img_name The image name
    @param img The image object
    @param coor The contour coordinates in form of [x1, y1, x2, y2, ...]
    @param bnd_box The bounding boxes in the image
    @param idx The image index
    @param category_id The category id
    
    Note:
    segmentation does not have area for now, 
    because it's not easy to compute we would add it if it is necessary
    '''
    label["images"].append({"file_name": img_name, "width": img.shape[1], "height": img.shape[0], "id": idx})
    label["annotations"].append({"id": idx, "image_id": idx, "category_id": category_id, 
                            "segmentation": [coor], 
                             "bbox": bnd_box, "iscrowd": 0})    


def write_coco_head(dscpt):
    label = {}
    label["info"] = {'version': None, 'description': dscpt}
    return label                 


def write_coco_tail(label, fileName):
    label_map_path = SEG_ANNO_DIR + '/mpotato_label_map.pbtxt'
    label["licenses"] = []
    label["categories"] = load_label_map(label_map_path)
    with open('{}.json'.format(os.path.join(SEG_ANNO_DIR, fileName)), 'w') as jsonFile:
        json.dump(label, jsonFile)


def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        content = f.read().splitlines()
        f.close()
    assert content is not None, 'cannot find label map'

    temp = list()
    for line in content:
        line = line.strip()
        if (len(line) > 2 and
                (line.startswith('id') or
                 line.startswith('name'))):
            temp.append(line.split(':')[1].strip())
    print(temp)

    label_dict = list()
    for idx in range(0, len(temp), 2):
        item_id = int(temp[idx])
        item_name = temp[idx + 1][1:-1]
        label_dict.append({'supercategory': 'food', 'id': item_id, 'name': item_name})
    return label_dict


def data_augmentation(img_name, idx):
    '''
    Data Augmentation on the segmented image. The file is from pytorch_pytorch_retinanet.retinanet_dataset.py
    All transformation method are import from transform.py, which are from pytorch_retinanet.utils
    
    Note:
    Right Now, it is randomly augment the image
    '''
    img = Image.open(os.path.join(SEG_IMG_DIR, img_name))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    size = 600 # the desired image size

    img = random_flip(img)
    img = random_crop(img)
    img = resize(img, (size, size))
    if random.random() > 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0, 1))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))
#         im1 = img.filter(ImageFilter.BLUR) # Gaussian Blur
    else:
        img = resize(img, (size, size))
        # img, boxes = center_crop(img, boxes, (size, size))

    filename = img_name[:-4]+"_"+str(idx)+".png"
    img.save(os.path.join(SEG_IMG_DIR + "/transformed", filename), "PNG")


def convert_to_coco(label, img_name, img, bboxes, img_idx, anno_idx, category_id):
    '''
    Write and save in COCO format.
    @param label The json directory object
    @param img_name The image name
    @param img The image object
    @param bnd_box The bounding boxes in the image
    @param idx The image index
    @param category_id The category id
    '''
    label["images"].append({"file_name": img_name, "width": img.shape[1], "height": img.shape[0], "id": img_idx})
    for bnd_box in bboxes:
        label["annotations"].append({"id": anno_idx, "image_id": img_idx, "category_id": category_id, 
                            "segmentation": [], 
                            "bbox": bnd_box, "iscrowd": 0})  
        anno_idx += 1

    return anno_idx


def choose_function(command):
    if command=="0": # crop image
        path = input("path: ")
        if not os.path.isdir(SEG_IMG_DIR):
            os.makedirs(SEG_IMG_DIR)
        image_filenames = sorted(os.listdir(os.path.join(path, 'images')))
        for img in image_filenames:
            if img[-4:]=='.png':
                xml = img[:-4]+'.xml'
                crop_img(path, img, xml)
                
    elif command=="1": # extract blue pixel
        if not os.path.isdir(SEG_IMG_DIR+"/obj"):
            os.makedirs(SEG_IMG_DIR+"/obj")
        seg_filenames = sorted(os.listdir(SEG_IMG_DIR))
        for img in seg_filenames:
            if img[-4:]=='.png':
                extract_object(img[:-4])
                
    elif command=="2": # create COCO label with contour
        if not os.path.isdir(SEG_ANNO_DIR):
            os.makedirs(SEG_ANNO_DIR)
        seg_filenames = sorted(os.listdir(SEG_IMG_DIR))
        dscpt = input("Description of Dataset: ")
        label = write_coco_head(dscpt)
        label["images"] = []
        label["annotations"] = []
        idx = 0
        for img in seg_filenames:
            if img[-4:]=='.png':
                img_file = cv2.imread(os.path.join(SEG_IMG_DIR, img))
                coor, bnd_box = detect_edge(img_file)
                write_coco_anno(label, img, img_file, coor, bnd_box, idx, 1)
                idx += 1
        write_coco_tail(label, "food")
                
    elif command=="3": # data augmentation on segmented image
        if not os.path.isdir(SEG_IMG_DIR+"/transformed"):
            os.makedirs(SEG_IMG_DIR+"/transformed")
        seg_filenames = sorted(os.listdir(SEG_IMG_DIR))
        idx = 0
        for img in seg_filenames:
            if img[-4:]=='.png':
                
                data_augmentation(img, idx)
                idx += 1
    
    elif command=="4": # move other food data
        assert os.path.isdir(ORI_IMG_DIR)
        seg_filenames = sorted(os.listdir(SEG_IMG_DIR))
        dscpt = input("Description of Dataset: ")
        label = write_json_head(dscpt)
        label["images"] = []
        label["annotations"] = []
        idx = 0
        for img in seg_filenames:
            if img[-4:]=='.png':
                split_img = img.split('+') # split the angle, trial, name
                angle = split_img[0].split('_')
                check_isolated = (angle[-1]=='isolated')
                if check_isolated: # background in blue
                    shutil.copy2(os.path.join(ORI_IMG_DIR, img), os.path.join(SEG_IMG_DIR, img))
                    img_file = cv2.imread(os.path.join(SEG_IMG_DIR, img))
                    coor, bnd_box = detect_edge(img_file)
                    
                    food_name = split_img[1].split('-')[0]
                    # label_dict hasn't loaded, need to change the code
                    write_json_anno(label, img, img_file, coor, bnd_box, idx, label_dict[food_name])
                    idx += 1
        write_json_tail(label, "food_train")
                
    elif command!="-1":
        print_usage()


def print_usage():
    print('Usage:')
    print('    python {} <keyword>\n'.format(sys.argv[0]))


def get_choice():
    print("crop segmented image 0")
    print("extract blue pixel 1")
    print("create contour label 2")
    print("data augmentation 3")
    print("other food 4")
    print("exit -1")
    return input(":")


if __name__=='__main__':
    command = 0
    while command!="-1":
        command = get_choice()
        choose_function(command)



#     elif command=="5": # Create COCO label from xml
#         path = "/Users/kyleghz/Desktop/data/val"
#         img_dir = os.path.join(path, 'images')
#         anno_dir = os.path.join(img_dir, 'annotations')
#         if not os.path.isdir(anno_dir):
#             os.makedirs(anno_dir)
#         dscpt = input("Description of Dataset: ")
#         label = write_coco_head(dscpt)
#         label["images"] = []
#         label["annotations"] = []
    
#         img_idx, anno_idx = 0, 0
#         image_filenames = sorted(os.listdir(img_dir))
#         for img in image_filenames:
#             if img[-4:]=='.png':
#                 xml = img[:-4]+'.xml'
#                 bboxes = crop_img(path, img, xml)
#                 img_file = cv2.imread(os.path.join(SEG_IMG_DIR, img))
#                 anno_idx = convert_to_coco(label, img, img_file, bboxes, img_idx, anno_idx, 1)
#                 img_idx += 1
#         write_coco_tail(label, "food")


# tree = ET.parse('/Users/kyleghz/Desktop/data/mashed_potato_dataset/xmls/potato-blue-trail-0_0000.xml')
# root = tree.getroot()
# bboxes = []


# for node in root:
#     if node.tag == 'object':
#         obj_name = node.find('name').text
#     if node.find('bndbox') is None:
#         continue
#     xmin = int(node.find('bndbox').find('xmin').text)
#     ymin = int(node.find('bndbox').find('ymin').text)
#     xmax = int(node.find('bndbox').find('xmax').text)
#     ymax = int(node.find('bndbox').find('ymax').text)# print(label['categories'])
#     bboxes.append([xmax, ymax, xmin, ymin])


# for bnd_box in bboxes:
#     print(bnd_box)

