import cxflow as cx
import numpy as np
import copy
import cv2

class MnistPredict(cx.AbstractHook):
    def after_batch(self, stream_name, batch_data):
        cv2.imwrite('./output/' + batch_data['id'] + 'b.jpeg', batch_data['images'][0, :, :, :])