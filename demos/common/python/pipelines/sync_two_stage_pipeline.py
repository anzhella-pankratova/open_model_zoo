"""
 Copyright (C) 2021 Intel Corporation
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

import logging
from models.utils import preprocess_output


class SyncTwoStagePipeline:
    def __init__(self, ie, encoder_model, decoder_model, en_device, de_device):
        self.logger = logging.getLogger()

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        self.logger.info('Loading encoder network to {} plugin...'.format(en_device))
        self.encoder_exec = ie.load_network(network=self.encoder_model.net, device_name=en_device)

        self.logger.info('Loading decoder network to {} plugin...'.format(de_device))
        self.decoder_exec = ie.load_network(network=self.decoder_model.net, device_name=de_device)

    def infer(self, frame):
        inputs, preprocessing_meta = self.encoder_model.preprocess(frame)
        raw_outputs = self.encoder_exec.infer(inputs)
        encoder_results = self.encoder_model.postprocess(raw_outputs, preprocessing_meta)

        decoder_results = []
        data = preprocess_output(encoder_results, frame)
        #print('boxes:', data)

        for inputs in data:
            inputs, preprocessing_meta = self.decoder_model.preprocess(inputs)
            raw_outputs = self.decoder_exec.infer(inputs)
            result = self.decoder_model.postprocess(raw_outputs, preprocessing_meta)[0]
            decoder_results.append(result)

        #print('encoder_results', encoder_results)
        #print('decoder results', decoder_results)
        return encoder_results, decoder_results