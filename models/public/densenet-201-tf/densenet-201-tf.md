# densenet-201-tf

## Use Case and High-Level Description

This is a TensorFlow\* version of `densenet-201` model, one of the DenseNet\* group of models designed to perform image classification.
For details, see [TensorFlow\* API docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201), [repository](https://github.com/tensorflow/tensorflow) and [paper](https://arxiv.org/abs/1608.06993).

## Specification

| Metric                          | Value           |
|---------------------------------|-----------------|
| Type                            | Classification  |
| GFlops                          | 8.6786          |
| MParams                         | 20.0013         |
| Source framework                | TensorFlow\*    |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 76.93%|
| Top 5  | 93.56%|

## Input

### Original Model

Image, name: `input_1` , shape: [1x224x224x3], format: [BxHxWxC],
where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: RGB.
Mean values - [123.68, 116.78, 103.94], scale values - [58.395,57.12,57.375].

### Converted Model

Image, name: `input_1`, shape: [1x3x224x224], [BxCxHxW],
where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: BGR.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `StatefulPartitionedCall/densenet201/predictions/Softmax`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted Model

The converted model has the same parameters as the original model.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tensorflow/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).