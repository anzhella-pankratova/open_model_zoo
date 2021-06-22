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
from collections import deque
from models.utils import preprocess_output
import numpy as np


class DecoderCell:
    def __init__(self, box, preprocessing_meta):
        self.status = 'IDLE' # IDLE -> BUSY, BUSY -> READY
        self.box = box
        self.preprocessing_meta = preprocessing_meta
        self.request_id = -100
        self.output = None


class EncoderCell:
    def __init__(self):
        self.reset_state()

    def update(self, id, request_id, preprocessing_meta, frame_meta):
        self.id = id
        self.request_id = request_id
        self.preprocessing_meta = preprocessing_meta
        self.frame_meta = frame_meta
        self.status = 'BUSY' # EMPTY -> BUSY, BUSY -> READY 

    def add_box(self, box, preprocessing_meta):
        cell = DecoderCell(box, preprocessing_meta=preprocessing_meta)
        self.decoder_cells.append(cell)

    def get_output(self):
        output = (self.output, self.frame_meta,
                  [cell.output for cell in self.decoder_cells])
        self.reset_state()
        return output

    def get_idle_decoder_cell(self):
        try:
            id = [cell.status for cell in self.decoder_cells].index('IDLE')
            return self.decoder_cells[id]
        except:
            return None

    def reset_state(self):
        self.id = -100
        self.request_id = None
        self.decoder_cells = []
        self.preprocessing_meta = None
        self.frame_meta = None
        self.output = None
        self.status = 'EMPTY'


class PerfectTwoStagePipeline:

    def __init__(self, ie, encoder, decoder, en_plugin_config, de_plugin_config,
                 en_device, de_device, en_num_requests, de_num_requests):
        self.logger = logging.getLogger()
        self.encoder = encoder
        self.decoder = decoder
        self.de_num_requests = de_num_requests

        self.logger.info('Loading encoder network to {} plugin...'.format(en_device))
        self.exec_net_encoder = ie.load_network(network=self.encoder.net, device_name=en_device,
                                                 config=en_plugin_config, num_requests=en_num_requests)
        self.logger.info('Loading decoder network to {} plugin...'.format(de_device))
        self.exec_net_decoder = ie.load_network(network=self.decoder.net, device_name=de_device,
                                                   config=de_plugin_config, num_requests=de_num_requests)

        self.encoder_cells = [EncoderCell() for _ in range(en_num_requests)]
        self.encoder_busy_req_id = []

        self.decoder_busy_req_id = []
        self.decoder_idle_req_id = deque([i for i in range(de_num_requests)])

        self.encoder_requests_map = {req_id: None for req_id in range(en_num_requests)}
        self.decoder_requests_map = {req_id: None for req_id in range(de_num_requests)}


    def get_result(self):
        # no detections case
        result = self.update_encoder_busy_to_ready()
        if result is not None:
            return result
        
        if 'READY' in [cell.status for cell in self.encoder_cells]:
            self.encoder_cells.sort(key=lambda x: x.id)

            #for _ in range(self.de_num_requests):
            self.update_decoder_idle_to_busy()
            #for _ in range(self.de_num_requests):
            self.update_decoder_busy_to_ready()

            return self.finish()

    def await_any(self):
        if self.decoder_busy_req_id:
            self.exec_net_decoder.wait(num_requests=min(len(self.decoder_idle_req_id) + 1, len(self.exec_net_decoder.requests)))
        elif self.encoder_busy_req_id:
            self.exec_net_encoder.wait(num_requests=min(len(self.exec_net_encoder.requests) -  len(self.encoder_busy_req_id) + 1,
                                       len(self.exec_net_encoder.requests)), timeout=1)

    def is_ready(self):
        for index, cell in enumerate(self.encoder_cells):
            if cell.status == 'EMPTY':
                return index
        return None

    def submit_data(self, inputs, cell_id, frame_id, frame_meta):
        inputs, preprocessing_meta = self.encoder.preprocess(inputs)
        req_id = self.exec_net_encoder.get_idle_request_id()

        self.encoder_cells[cell_id].update(frame_id, req_id, preprocessing_meta, frame_meta)
        self.encoder_requests_map[req_id] = self.encoder_cells[cell_id]

        self.encoder_busy_req_id.append(req_id)
        self.exec_net_encoder.requests[req_id].async_infer(inputs=inputs)

    def submit_data_to_decoder(self, cell):
        boxes = preprocess_output(cell.output, cell.frame_meta['frame'])
        for box in boxes:
            cell.add_box(*self.decoder.preprocess(box))
        return len(boxes) == 0

    def check_busy_requests(self, exec_net, busy_requests):
        req_id = exec_net.get_idle_request_id()
        if req_id == -1:
            return None
        if req_id in busy_requests:
            return req_id
        for id in busy_requests:
            if exec_net.requests[id].wait(0) == 0:
                return id

    def postprocess_result(self, net, exec_net, req_id, preprocessing_meta):
        request = exec_net.requests[req_id]
        raw_result = {key: blob.buffer for key, blob in exec_net.requests[req_id].output_blobs.items()}
        return net.postprocess(raw_result, preprocessing_meta)
    
    def update_encoder_busy_to_ready(self):
        req_id = self.check_busy_requests(self.exec_net_encoder, self.encoder_busy_req_id)
        if req_id is None: return

        cell = self.encoder_requests_map[req_id]
        cell.status = 'READY'
        cell.output = self.postprocess_result(self.encoder, self.exec_net_encoder,
                                              req_id, cell.preprocessing_meta)
        self.encoder_busy_req_id.remove(req_id)
        is_empty = self.submit_data_to_decoder(cell)
        if is_empty:
            return cell.get_output()

    def update_decoder_busy_to_ready(self):
        while True:
            req_id = self.check_busy_requests(self.exec_net_decoder, self.decoder_busy_req_id)
            if req_id is None: return

            cell = self.decoder_requests_map[req_id]
            cell.status = 'READY'
            cell.output = self.postprocess_result(self.decoder, self.exec_net_decoder,
                                                    req_id, cell.preprocessing_meta)
            self.decoder_busy_req_id.remove(req_id)
            self.decoder_idle_req_id.append(req_id)

    def update_decoder_idle_to_busy(self):
        if not self.decoder_idle_req_id: return

        while self.decoder_idle_req_id:
            req_id = self.decoder_idle_req_id.popleft()
            self.decoder_busy_req_id.append(req_id)

            for enc_cell in self.encoder_cells:
                dec_cell = enc_cell.get_idle_decoder_cell()
                if dec_cell is not None:
                    dec_cell.status = 'BUSY'
                    self.decoder_requests_map[req_id] = dec_cell

                    self.exec_net_decoder.requests[req_id].async_infer(inputs=dec_cell.box)
                    break
            if self.decoder_requests_map[req_id] is None:
                self.decoder_busy_req_id.remove(req_id)

    def finish(self):
        for enc_cell in self.encoder_cells:
            if set([cell.status for cell in enc_cell.decoder_cells]) != {'READY'}:
                continue
            return enc_cell.get_output()
        return None
