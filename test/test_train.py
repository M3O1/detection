import unittest
import os
import sys
import glob
import cv2
import numpy as np
sys.path.append("../")

from utils.utils import preprocess_input

image_dir = "../data/images/"


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

            mse = self.MSE(original_out, repaired_out)
            self.assertLessEqual(mse, 1e-5)

    def MSE(self, out1, out2):
        return np.sqrt(np.mean(np.square(out1 - out2)))

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

if __name__ == '__main__':
    unittest.main(verbosity=2)
