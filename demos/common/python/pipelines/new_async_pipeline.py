import logging
from collections import deque


class NewAsyncPipeline:
    def __init__(self, ie, model, plugin_config, device, max_num_requests):
        self.logger = logging.getLogger()
        self.model = model

        #self.logger.info('Loading network to {} plugin...'.format(device))
        self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                        config=plugin_config, num_requests=max_num_requests)
        self.num_requests = len(self.exec_net.requests)
        if max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.num_requests += 1
            self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                            config=plugin_config, num_requests=self.num_requests)
        requests_id = [id for id in range(self.num_requests)]
        self.empty_requests_id = deque(requests_id)
        self.busy_requests_id = deque([])
        self.model_meta = {req_id : None for req_id in requests_id}  # [frame_id, preprocess_meta, meta]
        self.results = {} # {frame_id: result}, it returns to the user

    def get_result(self, id):
        self.check_request_status()
        if id in self.results:
            return self.results.pop(id)
        return None

    def await_any(self):
        self.exec_net.wait(num_requests=1, timeout=1)

    def check_request_status(self):
        req_id = self.exec_net.get_idle_request_id()
        if req_id != -1 and req_id in self.busy_requests_id:
            result, id = self.get_detector_result(req_id)
            self.results[id] = result
            self.busy_requests_id.remove(req_id)
            self.empty_requests_id.append(req_id)

    def get_detector_result(self, request_id):
        request = self.exec_net.requests[request_id]
        frame_id, preprocess_meta, meta = self.get_model_meta(request_id)
        raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
        return (self.model.postprocess(raw_result, preprocess_meta), meta), frame_id

    def get_model_meta(self, request_id):
        meta = self.model_meta[request_id]
        self.model_meta[request_id] = None
        return meta

    def submit_data(self, inputs, id, meta):
        req_id = self.empty_requests_id.popleft()
        request = self.exec_net.requests[req_id]

        inputs, preprocessing_meta = self.model.preprocess(inputs)

        self.busy_requests_id.append(req_id)
        self.model_meta[req_id] = (id, preprocessing_meta, meta)

        request.async_infer(inputs=inputs)

    def is_ready(self):
        return len(self.empty_requests_id) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def has_completed_request(self):
        return len(self.busy_requests_id) > 0
