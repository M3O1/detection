import unittest
import os
import sys
import glob
import cv2
import numpy as np
from keras.models import load_model

sys.path.append("../")

from utils.utils import *

image_dir = "../data/images/"
model_path = "../data/raccoon.h5"

def MSE(out1, out2):
    return np.sqrt(np.mean(np.square(out1 - out2)))

class TestPreprocessInput(unittest.TestCase):
    def test_preprocess_input_in_all_images(self):
        net_h, net_w = 416, 416
        for imagename in os.listdir(image_dir):
            imagepath = os.path.join(image_dir, imagename)
            image = cv2.imread(imagepath)
            # original code
            original_out = self.preprocess_input_original(image, net_h, net_w)
            # repair code
            repaired_out = preprocess_input(image, net_h, net_w)

            mse = MSE(original_out, repaired_out)
            self.assertLessEqual(mse, 1e-5)

    def preprocess_input_original(self, image, net_h, net_w):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)//new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)//new_h
            new_h = net_h

        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image

class TestGetYoloBoxes(unittest.TestCase):
    def setUp(self):
        self.model = load_model(model_path)
        self.net_h = 416
        self.net_w = 416
        self.anchors = [17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281]
        self.obj_thresh = 0.5
        self.nms_thresh = 0.45

    def test_get_yolo_boxes_in_5_images(self):
        for imagename in os.listdir(image_dir)[:5]:
            imagepath = os.path.join(image_dir, imagename)
            image = cv2.imread(imagepath)
            # original code
            original_out = self.get_yolo_boxes_original(self.model, [image], self.net_h, self.net_w, self.anchors, self.obj_thresh, self.nms_thresh)
            # repair code
            repaired_out = get_yolo_boxes(self.model, [image], self.net_h, self.net_w, self.anchors, self.obj_thresh, self.nms_thresh)

            mse = MSE(self.boxes_to_arr(original_out),
                      self.boxes_to_arr(repaired_out))

            self.assertLessEqual(mse, 1e-5)

    def get_yolo_boxes_original(self, model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
        image_h, image_w, _ = images[0].shape
        nb_images           = len(images)
        batch_input         = np.zeros((nb_images, net_h, net_w, 3))

        # preprocess the input
        for i in range(nb_images):
            batch_input[i] = preprocess_input(images[i], net_h, net_w)

        # run the prediction
        batch_output = model.predict_on_batch(batch_input)
        batch_boxes  = [None]*nb_images

        for i in range(nb_images):
            yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
            boxes = []

            # decode the output of the network
            for j in range(len(yolos)):
                yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
                boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

            # correct the sizes of the bounding boxes
            correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

            # suppress non-maximal boxes
            do_nms(boxes, nms_thresh)

            batch_boxes[i] = boxes

        return batch_boxes

    def boxes_to_arr(self, boxes):
        arr = None
        for box in boxes:
            for b in box:
                b_arr = np.array([[int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax), b.get_label(), b.get_score()]])
                if arr is None:
                    arr = b_arr.copy()
                else:
                    arr = np.concatenate([arr, b_arr],axis=0)
        return arr

if __name__ == '__main__':
    unittest.main(verbosity=2)
