from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# 处理这种大小的图像。 请注意，这与 32 x 32 的原始 CIFAR 图像大小不同。
# 如果更改此数字，则整个模型体系结构将发生变化，任何模型都需要重新训练。
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
# 描述 CIFAR-10 数据集的全局常数。
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
     从 filename_queue 中读取 CIFAR10 二进制数据，构造成样本数据

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  # cifar10 的数据集共有 6 万幅 32 * 32 大小的图片，分为 10 类，每类 6000 张，其中 5 万张用于训练， 1 万张用于测试。
  # 数据集被分成了 5 个训练的 batches (data_batch_1.bin ~ data_batch_5.bin) 和 1 个测试的 batch (test_batch.bin)。每个 batch 里的图片都是随机排列的。
  # 每个 bin 文件的格式如下：
  #
  # <1 x label><3072 x pixel>
  # ...
  # <1 x label><3072 x pixel>
  #
  # 共有一万行，每行 3073 个字节，第一个字节表示标签信息，剩下的 3072 字节分为 RGB 三通道，每个通道 1024( = 32 * 32) 个字节。
  # 注意，行与行之间没有明显的区分标识符，所以整个 bin 文件字节长度恰好是 3073 万。
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  # 每个记录都包含标签信息和图片信息，每个记录都有固定的字节数（3073 = 1 + 3072）。
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  # 从 filename_queue 获取文件名，读取记录。
  # CIFAR-10 文件中没有页眉或页脚，所以我们把 header_bytes 和 footer_bytes 设置为默认值0。

  # TensorFlow 使用 tf.FixedLengthRecordReader 读取固定长度格式的数据，与 tf.decode_raw 配合使用
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # 从一个字符串转换为一个 uint8 的向量，即 record_bytes 长。
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  # 采用 tf.strided_slice 方法在 record_bytes 中提取第一个 bytes 作为标签，从 uint8 转换为 int32。
  # tf.slice(record_bytes, 起始位置， 长度)
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # 记录中标签后的剩余字节代表图像，从 label 起，在 record_bytes 中提取 self.image_bytes = 3072 长度为图像，
  # 从 [depth * height * width] 转化为 [depth，height，width]，图片转化成 3*32*32。
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  # 从 [depth, height, width] 转化为 [height, width, depth]，图片转化成 32*32*3。
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
     构造 batch_size 样本集
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
      在队列中保留的最小样本数量。
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
    shuffle 的作用在于指定是否需要随机打乱样本的顺序，一般作用于训练阶段，提高鲁棒性。

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  # 创建一个随机打乱样本顺序的队列，然后从示例队列中读取 batch_size 个图像+标签
  num_preprocess_threads = 16
  if shuffle:
  # 当 shuffle = true 时，每次从队列中 dequeue 取数据时，不再按顺序，而是随机的，所以打乱了样本的原有顺序。
  # shuffle 还要配合参数 min_after_dequeue 使用才能发挥作用。
  # 这个参数 min_after_dequeue 的意思是队列中，做 dequeue（取数据）的操作后，queue runner 线程要保证队列中至少剩下 min_after_dequeue 个数据。
  # 如果 min_after_dequeue 设置的过少，则即使 shuffle 为 true，也达不到好的混合效果。
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
  # 当 shuffle = false 时，每次 dequeue 是从队列中按顺序取数据，遵从先入先出的原则
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # 在可视化器中显示训练图像。
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


"""
  原始图片经过了部分预处理之后，才送入模型进行训练或评估。
  原始的图片尺寸为32*32的像素尺寸，主要的预处理是两步:
  1、 首先将其裁剪为24*24像素大小的图片，其中训练集是随机裁剪，测试集是沿中心裁
  2、 将图片进行归一化，变为0均值，1方差
  其中为了增加样本量，我们还对训练集增加如下的预处理:
  1、 随机的对图片进行由左到右的翻转
  2、 随机的改变图片的亮度
  3、 随机的改变图片的对比度
  4、 最后是图片的白化
"""
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
     使用 Reader ops 将样本数据进行预处理，构造成 CIFAR 训练数据

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  #Debug: change .bin
  #filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
  filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 生成要读取的文件名队列
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.
  # 为训练网络进行图像处理。注意应用于图像的许多随机失真。

  # Randomly crop a [height, width] section of the image.
  # 随机裁剪图像为 [height，width] 像素大小的图片
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  # 随意地水平翻转图像。
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # 因为这些操作是不可交换的，所以请考虑将它们的操作随机化。
  # 随机的改变图片的亮度
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  # 随机的改变图片的对比度
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  # 图像的白化：减去平均值并除以像素的方差，均值与方差的均衡，降低图像明暗、光照差异引起的影响
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  # 确保随机 shuffling 具有良好的混合性能。
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  # 构造 batch_size 样本集（图像+标签）
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
     使用 Reader ops 将样本数据进行预处理，构造成 CIFAR 测试数据构建
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 生成要读取的文件名队列
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  # 从文件名队列中的文件读取示例
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  # 通过构建一个示例队列生成一批图像和标签。
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)



# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""
""" 用于解码 CIFAR-10 二进制文件格式的例程。"""
