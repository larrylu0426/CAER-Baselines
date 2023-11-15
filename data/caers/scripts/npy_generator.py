import argparse
import copy
import os
import pathlib

import cv2
import numpy as np
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


def prepare_data(data, data_dir, save_dir, phase):
    face_arr = list()
    context_arr = list()
    img_arr = list()
    bbox_arr = list()
    label_arr = list()

    for _, (item) in tqdm(enumerate(data), total=len(data), leave=True):
        try:
            sample = item.split(',')
            path = os.path.join(data_dir, sample[0])

            label = int(sample[1])
            x1, y1, x2, y2 = int(sample[2]), int(sample[3]), int(
                sample[4]), int(sample[5])
            img = Image.open(path)
            W, H = img.size
            face = img.crop((x1, y1, x2, y2))
            context = copy.deepcopy(img)
            draw = ImageDraw.Draw(context)
            draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
            bbox = np.array([x1, y1, x2, y2])
            # resize accrording to description in the paper: "facial are resized to have the frame size of 96 × 96, the clips of contextual parts are resized to have the frame size of 128(H) × 171(W)"
            face = transforms.Resize((96, 96))(face)
            context = transforms.Resize((128, 171))(context)
            img = transforms.Resize((128, 171))(img)
            # # PIL to OpenCV
            face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
            context = cv2.cvtColor(np.array(context), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bbox = resize_bbox(bbox, (H, W), (128, 171))

            face_arr.append(face)
            context_arr.append(context)
            img_arr.append(img)
            bbox_arr.append(bbox)
            label_arr.append(label)
        except Exception as e:
            print(path)
            raise e
    face_arr = np.array(face_arr)
    context_arr = np.array(context_arr)
    img_arr = np.array(img_arr)
    bbox_arr = np.array(bbox_arr)
    label_arr = np.array(label_arr)

    print(len(data), face_arr.shape, context_arr.shape)
    np.save(os.path.join(save_dir, '%s_face_arr.npy' % (phase)), face_arr)
    np.save(os.path.join(save_dir, '%s_context_arr.npy' % (phase)),
            context_arr)
    np.save(os.path.join(save_dir, '%s_img_arr.npy' % (phase)), img_arr)
    np.save(os.path.join(save_dir, '%s_bbox_arr.npy' % (phase)), bbox_arr)
    np.save(os.path.join(save_dir, '%s_label_arr.npy' % (phase)), label_arr)
    print(face_arr.shape, context_arr.shape, img_arr.shape, bbox_arr.shape,
          label_arr.shape)
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

    phases = ['train', 'test']

    print('loading Annotations')
    for phase in phases:
        data = [
            line.removesuffix("\n")
            for line in open(os.path.join(PROJECT_DIR, '{}.txt'.format(phase)),
                             'r').readlines()
        ]
        print('starting phase ', phase)
        prepare_data(data, args.data_dir, args.save_dir, phase)
