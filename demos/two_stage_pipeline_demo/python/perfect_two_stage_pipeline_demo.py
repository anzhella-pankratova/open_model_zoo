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
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


import models
import monitors
from images_capture import open_images_capture
from performance_metrics import PerformanceValues

from pipelines import get_user_config, PerfectTwoStagePipeline

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m_fd', required=True, type=Path,
                      help='Required. Path to an .xml file with a Face Detection model.')
    args.add_argument('-m_ld', required=True, type=Path,
                      help='Required. Path to an .xml file with a Landmakrs Detection model.')
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d_fd', '--device_fd', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on Detection stage; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('-d_ld', '--device_ld', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on Recognition stage; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.8, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                                   help='Optional. Keeps aspect ratio on resize.')
                                   
    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq_fd', '--num_requests_fd', default=1, type=int,
                            help='Optional. Number of infer requests for Face Detection stage.')
    infer_args.add_argument('-nstreams_fd', '--num_streams_fd',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>) for Detection stage.',
                            default='', type=str)
    infer_args.add_argument('-nthreads_fd', '--num_threads_fd', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) '
                                 'for Detection stage.')
    infer_args.add_argument('-nstreams_ld', '--num_streams_ld',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>) for Recognition stage.',
                            default='', type=str)
    infer_args.add_argument('-nthreads_ld', '--num_threads_ld', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) '
                                 'for Recognition stage.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output',
                         help='Optional. Name of output to save.')
    io_args.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    return parser


def draw_detections(frame, detections, landmarks):
    color = (50, 205, 50)
    face_id = 0
    for _, detection in enumerate(detections):
        if detection.score > 0.8:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), frame.shape[1])
            ymax = min(int(detection.ymax), frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            if len(landmarks):
                for point in landmarks[face_id][0]:
                    x = xmin + (xmax - xmin) * point[0]
                    y = ymin + (ymax - ymin) * point[1]
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
            face_id += 1
    return frame


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config_fd = get_user_config(args.device_fd, args.num_streams_fd, args.num_threads_fd)
    plugin_config_ld = get_user_config(args.device_ld, args.num_streams_ld, args.num_threads_ld)

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    frame_size = frame.shape[:2]
    input_transform = models.InputTransform(False, None, None)
    detector = models.SSD(ie, args.m_fd, input_transform, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
    landmarker = models.LandmarksDetector(ie, args.m_ld)

    encoder_requests = 2 if not args.num_streams_ld else int(args.num_streams_ld) * 2
    pipeline = PerfectTwoStagePipeline(ie, detector, landmarker,
                                       plugin_config_fd, plugin_config_ld,
                                       args.device_fd, args.device_ld,
                                       args.num_requests_fd, encoder_requests)

    FRAMES_NUM = 300
    counter = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    pipeline.submit_data(frame, {'frame': frame, 'start_time': start_time})

    next_frame_id = 1
    next_frame_id_to_show = 0

    perf_values = PerformanceValues(10)
    video_writer = cv2.VideoWriter()
    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame_size[1] / 4), round(frame_size[0] / 8)))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame_size[1], frame_size[0])):
        raise RuntimeError("Can't open video writer")

    while True:
        results = pipeline.get_result()
        if results:
            detections, frame_meta, landmarks = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            frame = draw_detections(frame, detections, landmarks)
            perf_values.update(start_time)

            if counter <= FRAMES_NUM:
                counter += 1
            else:
                break

            next_frame_id_to_show += 1
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Perfect Two Stage Pipeline Results', frame)
                key = cv2.waitKey(1)
                if key in {ord('q'), ord('Q'), 27}:
                    break
                presenter.handleKey(key)
            continue

        if pipeline.is_ready():
            # Get new frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break
            # Submit to face detection model
            pipeline.submit_data(frame, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        #else:
        #   pipeline.await_any()

    latency, fps = perf_values.get_total()

    with open('perfect_two_stage_pipeline_results.txt', 'a', encoding='utf8') as f:
        print("\nInput: {}".format(args.input), file=f)
        print("Face Detection model: {}\nLandmarks detection model: {}".format(args.m_fd, args.m_ld), file=f)
        print("Parameters: nireq_fd {}, nstreams_fd {}, nireq_ld {}, nstreams_ld {}".format(
            args.num_requests_fd, args.num_streams_fd, encoder_requests, args.num_streams_ld), file=f)
        print("Mode: {}".format(pipeline.mode), file=f)
        print("Latency: {} ms".format(np.round(latency, 3)), file=f)
        print("FPS: {}".format(np.round(fps, 3)), file=f)

    print("Mode:", pipeline.mode)
    print("Request.wait calls:", pipeline.wait_calls)
    print("Latency: {} ms".format(np.round(latency, 3)))
    print("FPS: {}".format(np.round(fps, 3)))


if __name__ == '__main__':
    sys.exit(main() or 0)
