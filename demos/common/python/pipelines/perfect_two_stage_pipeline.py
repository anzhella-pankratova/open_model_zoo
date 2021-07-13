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

from collections import deque
from models.utils import preprocess_output


class DecoderCell:
    def __init__(self, box, preprocessing_meta, parent):
        self.parent = parent
        self.box = box
        self.preprocessing_meta = preprocessing_meta
        self.output = None


class EncoderCell:
    def __init__(self, request, preprocessing_meta, frame_meta):
        self.request = request
        self.preprocessing_meta = preprocessing_meta
        self.frame_meta = frame_meta

        self.decoder_cells = []
        self.is_ready = False
        self.output = None

        self.total_boxes = 0
        self.ready_boxes = 0
        self.next_processed = 0

    def create_decoder_cells(self, decoder):
        boxes = preprocess_output(self.output, self.frame_meta['frame'])
        self.decoder_cells = [DecoderCell(*decoder.preprocess(box), parent=self) for box in boxes]
        self.total_boxes = len(self.decoder_cells)
        if self.total_boxes == 0:
            self.is_ready = True

    def get_output(self):
        return (self.output, self.frame_meta, [cell.output for cell in self.decoder_cells])

    def get_next_decoder_cell(self):
        if not self.decoder_cells:
            return
        next_id = self.next_processed
        if next_id < self.total_boxes:
            self.next_processed += 1
            return self.decoder_cells[next_id]

    def check_is_ready(self):
        if self.ready_boxes != 0 and self.ready_boxes == self.total_boxes:
            self.is_ready = True


class PerfectTwoStagePipeline:

    def __init__(self, ie, encoder, decoder, en_plugin_config, de_plugin_config,
                 en_device, de_device, en_num_requests, de_num_requests):

        self.encoder, self.decoder = encoder, decoder

        self.exec_encoder = ie.load_network(network=self.encoder.net, device_name=en_device,
                                            config=en_plugin_config, num_requests=en_num_requests)
        if en_num_requests == 0:
            self.exec_encoder = ie.load_network(network=self.encoder.net, device_name=en_device,
                                                config=en_plugin_config, num_requests=len(self.exec_encoder.requests) + 1)
        self.exec_decoder = ie.load_network(network=self.decoder.net, device_name=de_device,
                                            config=de_plugin_config, num_requests=de_num_requests)
        if de_num_requests == 0:
            self.exec_decoder = ie.load_network(network=self.decoder.net, device_name=de_device,
                                                config=de_plugin_config, num_requests=len(self.exec_decoder.requests) + 1)

        self.encoder_cells = deque([])

        self.encoder_busy_requests = []
        self.decoder_busy_requests = []
        self.decoder_idle_requests = deque(self.exec_decoder.requests)

        self.encoder_requests_map = {request: None for request in self.exec_encoder.requests}
        self.decoder_requests_map = {request: None for request in self.exec_decoder.requests}

    def get_result(self):
        if not self.encoder_cells:
            return
        self.encoder_update_busy_to_ready()

        if self.encoder_cells[0].is_ready:
            return self.encoder_cells.popleft().get_output()

        self.decoder_update_idle_to_busy()
        self.decoder_update_busy_to_ready()

        for cell in self.encoder_cells:
            cell.check_is_ready()

    def is_ready(self):
        return len(self.encoder_cells) < len(self.exec_encoder.requests)

    def submit_data(self, inputs, frame_meta):
        inputs, preprocessing_meta = self.encoder.preprocess(inputs)

        request = self.exec_encoder.requests[self.exec_encoder.get_idle_request_id()]
        request.async_infer(inputs)

        cell = EncoderCell(request, preprocessing_meta, frame_meta)
        self.encoder_cells.append(cell)
        self.encoder_requests_map[request] = cell
        self.encoder_busy_requests.append(request)

    def get_processed_request(self, exec_net, busy_requests):
        req_id = exec_net.get_idle_request_id()
        if req_id == -1:
            return None
        request = exec_net.requests[req_id]
        if request in busy_requests:
            return request
        for request in busy_requests:
            if request.wait(0) == 0:
                return request

    def postprocess_result(self, net, request, preprocessing_meta):
        raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
        return net.postprocess(raw_result, preprocessing_meta)

    def encoder_update_busy_to_ready(self):
        while self.encoder_busy_requests:
            request = self.get_processed_request(self.exec_encoder, self.encoder_busy_requests)
            if request is None:
                return

            cell = self.encoder_requests_map[request]
            cell.output = self.postprocess_result(self.encoder, request, cell.preprocessing_meta)
            self.encoder_busy_requests.remove(request)

            cell.create_decoder_cells(self.decoder)

    def decoder_update_busy_to_ready(self):
        while self.decoder_busy_requests:
            request = self.get_processed_request(self.exec_decoder, self.decoder_busy_requests)
            if request is None:
                return
            cell = self.decoder_requests_map[request]
            cell.output = self.postprocess_result(self.decoder, request, cell.preprocessing_meta)
            self.decoder_busy_requests.remove(request)

            cell.parent.ready_boxes += 1
            self.decoder_idle_requests.append(request)

    def decoder_update_idle_to_busy(self):
        while self.decoder_idle_requests:
            cell = self.get_decoder_cell_to_process()
            if cell is None:
                return
            request = self.decoder_idle_requests.popleft()
            request.async_infer(cell.box)

            self.decoder_busy_requests.append(request)
            self.decoder_requests_map[request] = cell

    def get_decoder_cell_to_process(self):
        for encoder_cell in self.encoder_cells:
            decoder_cell = encoder_cell.get_next_decoder_cell()
            if decoder_cell:
                return decoder_cell
