import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        print(self.images_path)
        iou_arr = []
        iou_arr1 = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.Detector()

        # My detector
        import detectors.my_super_detector.mydetector as my_detector
        my_detector = my_detector.MyDetector()

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            #img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            #img = preprocess.resize_image(img)
            img = preprocess.sharpening(img)
            img = preprocess.denoising(img)

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)
            my_prediction_list = my_detector.detect(im_name.split(os.sep)[3])

            print(im_name.split(os.sep)[3])

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # Only for detection:
            # p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            #
            # iou = eval.iou_compute(p, gt)
            #
            # iou_arr.append(iou)

            p1, gt1 = eval.prepare_for_detection(my_prediction_list, annot_list)

            iou1 = eval.iou_compute(p1, gt1)

            iou_arr1.append(iou1)

      #  miou = np.average(iou_arr)
        print("\n")
        # print("Average IOU(iou_arr):", f"{miou:.2%}")
        miou1 = np.average(iou_arr1)
        print("\n")
        print("Average IOU(iou_arr1):", f"{miou1:.2%}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()