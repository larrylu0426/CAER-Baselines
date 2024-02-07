import argparse
import ast
import copy
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

PROJECT_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent


def resize_bbox(bbox, origin_size, target_size):
    oirgin_H, origin_W = origin_size
    target_H, target_W = target_size
    resize_y = target_H / oirgin_H
    resize_x = target_W / origin_W
    n_bbox = np.array([
        resize_x * bbox[0], resize_y * bbox[1], resize_x * bbox[2],
        resize_y * bbox[3]
    ])
    return n_bbox


def one_hot(cats):
    ret = np.zeros(26)
    cat_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', \
                'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
                'Disquietment', 'Doubt/Confusion', 'Embarrassment', \
                'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',\
                'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', \
                'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    for cat in cats:
        ret[cat_labels.index(cat)] = 1
    return ret


def prepare_data(data, data_dir, save_dir, phase):
    body_arr = list()
    context_arr = list()
    img_arr = list()
    bbox_arr = list()
    cat_label_arr = list()
    cont_label_arr = list()

    for _, (item) in tqdm(data.iterrows(), total=len(data), leave=True):
        try:
            path = os.path.join(data_dir, item['Folder'], item['Filename'])
            cat_label = one_hot(ast.literal_eval(item['Categorical_Labels']))
            cont_label = np.float32(ast.literal_eval(
                item['Continuous_Labels'])) / 10.0
            x1, y1, x2, y2 = [int(i) for i in ast.literal_eval(item['BBox'])]
            img = Image.open(path)
            W, H = img.size
            body = img.crop((x1, y1, x2, y2))
            context = copy.deepcopy(img)
            draw = ImageDraw.Draw(context)
            try:
                # color image
                draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
            except:
                # gray image
                draw.rectangle((x1, y1, x2, y2), fill=(0))
            bbox = np.array([x1, y1, x2, y2])
            body = transforms.Resize((128, 128))(body)
            context = transforms.Resize((224, 224))(context)
            img = transforms.Resize((224, 224))(img)
            # PIL to OpenCV
            body = cv2.cvtColor(np.array(body), cv2.COLOR_RGB2BGR)
            context = cv2.cvtColor(np.array(context), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bbox = resize_bbox(bbox, (H, W), (224, 224))

            body_arr.append(body)
            context_arr.append(context)
            img_arr.append(img)
            bbox_arr.append(bbox)
            cat_label_arr.append(cat_label)
            cont_label_arr.append(cont_label)
        except Exception as e:
            print(path)
            raise e
    body_arr = np.array(body_arr)
    context_arr = np.array(context_arr)
    img_arr = np.array(img_arr)
    bbox_arr = np.array(bbox_arr)
    cat_label_arr = np.array(cat_label_arr)
    cont_label_arr = np.array(cont_label_arr)

    print(len(data), body_arr.shape, context_arr.shape)
    np.save(os.path.join(save_dir, '%s_body_arr.npy' % (phase)), body_arr)
    np.save(os.path.join(save_dir, '%s_context_arr.npy' % (phase)),
            context_arr)
    np.save(os.path.join(save_dir, '%s_img_arr.npy' % (phase)), img_arr)
    np.save(os.path.join(save_dir, '%s_bbox_arr.npy' % (phase)), bbox_arr)
    np.save(os.path.join(save_dir, '%s_cat_label_arr.npy' % (phase)),
            cat_label_arr)
    np.save(os.path.join(save_dir, '%s_cont_label_arr.npy' % (phase)),
            cont_label_arr)
    print(body_arr.shape, context_arr.shape, img_arr.shape, bbox_arr.shape,
          cat_label_arr.shape, cont_label_arr.shape)
    print('completed generating %s data files' % (phase))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory name in which data will be preprocessed')
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory name in which preprocessed data will be stored')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    phases = ['train', 'val', 'test']

    print('loading Annotations')
    for phase in phases:
        data = pd.read_csv(os.path.join(PROJECT_DIR, '{}.csv'.format(phase)))
        print('starting phase ', phase)
        prepare_data(data, args.data_dir, args.save_dir, phase)
