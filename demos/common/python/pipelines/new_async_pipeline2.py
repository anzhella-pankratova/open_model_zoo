import logging
from collections import deque
from models.utils import preprocess_output


class NewAsyncPipeline2:
    def __init__(self, ie, detector, td_plugin_config, td_device, td_num_requests):
        self.logger = logging.getLogger()
        self.detector = detector

        self.logger.info('Loading Text Detection network to {} plugin...'.format(td_device))
        self.exec_net_detector = ie.load_network(network=self.detector.net, device_name=td_device,
                                                 config=td_plugin_config, num_requests=td_num_requests)

        self.td_num_requests = td_num_requests
        detector_req_id = [id for id in range(td_num_requests)]

        self.empty_detector_req_id = deque(detector_req_id)
        self.processed_detector_req_id = deque([])
        self.detector_meta = {req_id : None for req_id in detector_req_id}     # [frame_id, preprocess_meta, meta]
        self.detector_result = {} # {frame_id: result}, it returns to the user


    def get_result(self, id):
        self.check_detector_status()

        if id in self.detector_result:
            return self.detector_result.pop(id)

        return None


    def await_any(self):
        if self.processed_detector_req_id:
            self.exec_net_detector.wait(num_requests=min(len(self.empty_detector_req_id) + 1, len(self.exec_net_detector.requests)), timeout=1)

    def check_detector_status(self):
        req_id = self.exec_net_detector.get_idle_request_id()
        if req_id == -1:
            self.await_any()
            req_id = self.exec_net_detector.get_idle_request_id()
        if req_id in self.processed_detector_req_id:
            result, id = self.get_detector_result(req_id)
            self.detector_result[id] = result
            self.processed_detector_req_id.remove(req_id)
            self.empty_detector_req_id.append(req_id)

    def get_detector_result(self, request_id):
        request = self.exec_net_detector.requests[request_id]
        frame_id, preprocess_meta, meta = self.get_detector_meta(request_id)
        raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
        return (self.detector.postprocess(raw_result, preprocess_meta), meta), frame_id

    def get_detector_meta(self, request_id):
        meta = self.detector_meta[request_id]
        self.detector_meta[request_id] = None
        return meta

    def submit_data(self, inputs, id, meta):
        request_id = self.empty_detector_req_id.popleft()
        request = self.exec_net_detector.requests[request_id]

        inputs, preprocessing_meta = self.detector.preprocess(inputs)

        self.processed_detector_req_id.append(request_id)
        self.detector_meta[request_id] = (id, preprocessing_meta, meta)

        request.async_infer(inputs=inputs)

    def is_ready(self):
        return len(self.empty_detector_req_id) != 0

    def await_all(self):
        for request in self.exec_net_detector.requests:
            request.wait()

    def has_completed_request(self):
        return len(self.processed_detector_req_id) > 0
