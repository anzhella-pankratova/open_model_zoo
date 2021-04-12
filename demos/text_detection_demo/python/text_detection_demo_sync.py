#!/usr/bin/env python3
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
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m_td', required=True, type=Path,
                      help='Required. Path to an .xml file with a Text Detection model.')
    args.add_argument('-m_tr', required=True, type=Path,
                      help='Required. Path to an .xml file with a Text Recognition model.')
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output',
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-t', '--prob_threshold', default=0.7, type=float,
                       help='Optional. Probability threshold for text detections filtering.')
    args.add_argument('-b', '--bandwidth', default=0, type=int,
                      help='Optional. Bandwidth for CTC beam search decoder. Default value is 0, '
                           'in this case CTC greedy decoder will be used.')
    args.add_argument('-l', '--cpu_extension',
                      help='Optional. For CPU custom layers, if any. Absolute path to a shared library with the '
                           'kernels implementation.', type=str, default=None)
    args.add_argument('-d', '--device',
                      help='Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for the device specified. '
                           'Default value is CPU.',
                      default='CPU', type=str)
    args.add_argument('--no_show', action='store_true', help="Optional. Don't show output.")
    args.add_argument('-u', '--utilization-monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')

    return parser


def draw_detections(frame, detections, labels):
    color = (50, 205, 50)
    for detection in detections:
        xmin = max(int(detection.xmin), 0)
        ymin = max(int(detection.ymin), 0)
        xmax = min(int(detection.xmax), frame.shape[1])
        ymax = min(int(detection.ymax), frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    for idx, detection in enumerate(detections):
        xmin = max(int(detection.xmin), 0)
        ymin = max(int(detection.ymin), 0)
        label = labels[idx]
        textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return frame


if __name__ == '__main__':
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    log.info('Loading Text Detection network...')
    text_detection = models.CTPN(ie, args.m_td, input_size=frame.shape[:2], threshold=args.prob_threshold)
    text_recognition = models.TextRecognition(ie, args.m_tr, alphabet='0123456789abcdefghijklmnopqrstuvwxyz#', bandwidth=args.bandwidth)
    
    log.info('Loading Text Recognition network...')
    text_detection_exec = ie.load_network(network=text_detection.net, device_name=args.device)
    text_recognition_exec = ie.load_network(network=text_recognition.net, device_name=args.device)

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    frames_processed = 0
    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()
    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    total_latency = 0
    total_fps = 0
    counter = 0

    while frame is not None:
        inputs, meta = text_detection.preprocess(frame)
        outputs = text_detection_exec.infer(inputs=inputs)
        detections = text_detection.postprocess(outputs, meta)

        texts = []
        for detection in detections:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), frame.shape[1])
            ymax = min(int(detection.ymax), frame.shape[0])
            cropped_frame = frame[ymin:ymax, xmin:xmax]

            inputs, meta = text_recognition.preprocess(cropped_frame)
            outputs = text_recognition_exec.infer(inputs=inputs)
            text = text_recognition.postprocess(outputs, meta)
            texts.append(text)

        frame = draw_detections(frame, detections, texts)
        presenter.drawGraphs(frame)
        metrics.update(start_time, frame)

        if counter <= 500:
            latency, fps = metrics.get_total()
            if latency and fps:
                total_latency += latency
                total_fps += fps
                counter += 1
        else:
            break

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Text detection Demo', frame)
            key = cv2.waitKey(1)
            if key in {ord("q"), ord("Q"), 27}:
                break
            presenter.handleKey(key)
        
        start_time = perf_counter()
        frame = cap.read()

    print('Mean metrics for 200 frames')
    print("Total Latency: {:.1f} ms".format(total_latency * 1e3 / 200))
    print("FPS: {:.1f}".format(total_fps / 200))
    #metrics.print_total()
    #print(presenter.reportMeans())
