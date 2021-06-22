"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from .model import Model
from .utils import preprocess_output, resize_image


class LandmarksDetector(Model):
    POINTS_NUMBER = 5

    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)

        assert len(self.net.input_info) == 1, 'Expected 1 input blob'
        assert len(self.net.outputs) == 1, 'Expected 1 output blob'
        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape
        output_shape = self.net.outputs[self.output_blob].shape

        assert np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1], output_shape), \
            'Expected model output shape {}, got {}'.format([1, self.POINTS_NUMBER * 2, 1, 1], output_shape)

    def resize_input(self, image, target_shape):
        #print(image.shape)
        _, _, h, w = target_shape
        resized_image = resize_image(image, (w, h))
        #print(resized_image.shape)
        resized_image = resized_image.transpose((2, 0, 1)) # HWC->CHW
        resized_image = resized_image.reshape(target_shape)
        return resized_image
    
    def preprocess(self, inputs):
        boxes = self.resize_input(inputs, self.input_shape)
        dict_inputs = {self.input_blob: boxes}
        return dict_inputs, {}

    def postprocess(self, outputs, meta):
        outputs = outputs[self.output_blob]
        return [out.reshape((-1, 2)).astype(np.float64) for out in outputs]
