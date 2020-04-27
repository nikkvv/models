# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""tf.data.Dataset builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
import functools
import tensorflow as tf

from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2


def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def read_dataset(file_read_func, input_files, config):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
      read every individual file into a tf.data.Dataset.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.gfile.Glob(input_files)
  num_readers = config.num_readers
  if num_readers > len(filenames):
    num_readers = len(filenames)
    tf.logging.warning('num_readers has been reduced to %d to match input file '
                       'shards.' % num_readers)
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if config.shuffle:
    filename_dataset = filename_dataset.shuffle(
        config.filenames_shuffle_buffer_size)
  elif num_readers > 1:
    tf.logging.warning('`shuffle` is false, but the input data stream is '
                       'still slightly shuffled since `num_readers` > 1.')
  filename_dataset = filename_dataset.repeat(config.num_epochs or None)
  records_dataset = filename_dataset.apply(
      tf.contrib.data.parallel_interleave(
          file_read_func,
          cycle_length=num_readers,
          block_length=config.read_block_length,
          sloppy=config.shuffle))
  if config.shuffle:
    records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
  return records_dataset


############################
##### NIKHIL Start #########
############################
def xml_parse_function(filename):
  # Read xml file
  image_string = tf.read_file(filename)

  # Get image name from xml
  image_name = tf.strings.split([image_string], sep='<path>')
  image_name = image_name.values[-1]
  image_name = tf.strings.split([image_name], sep='</path>')
  image_name = image_name.values[0]

  # Get image_boxes
  # tf.split() return SparseTensor. Therefore using .values
  image_objects = tf.strings.split([image_string], sep='<object>')

  # Total number of bounding boxes in xml
  num_objects = tf.cond(tf.greater(tf.shape(image_objects.values)[0], 1), 
                                    lambda: tf.shape(image_objects.values)[0] - 1,
                                    lambda: 0)

  # Getting all the image objects from xml
  image_objects = tf.cond(tf.greater(tf.shape(image_objects.values)[0], 1), 
                          lambda: image_objects.values[1:], 
                          lambda: tf.as_string(0))

  # Function used to map from <object> in sml to a bounding box [class, xmin, ymin, xmax, ymax]
  def get_box_class_and_loc(obj):
    # <name>Image File Path</name>
    obj_class = tf.strings.split([obj], sep='<name>')
    obj_class = obj_class.values[-1]
    obj_class = tf.strings.split([obj_class], sep='</name>')
    obj_class = obj_class.values[0]

    # <xmin>688</xmin>
    # <ymin>386</ymin>
    # <xmax>785</xmax>
    # <ymax>439</ymax>
    obj_xmin = tf.strings.split([obj], sep='<xmin>')
    obj_xmin = obj_xmin.values[-1]
    obj_xmin = tf.strings.split([obj_xmin], sep='</xmin>')
    obj_xmin = obj_xmin.values[0]

    obj_ymin = tf.strings.split([obj], sep='<ymin>')
    obj_ymin = obj_ymin.values[-1]
    obj_ymin = tf.strings.split([obj_ymin], sep='</ymin>')
    obj_ymin = obj_ymin.values[0]

    obj_xmax = tf.strings.split([obj], sep='<xmax>')
    obj_xmax = obj_xmax.values[-1]
    obj_xmax = tf.strings.split([obj_xmax], sep='</xmax>')
    obj_xmax = obj_xmax.values[0]

    obj_ymax = tf.strings.split([obj], sep='<ymax>')
    obj_ymax = obj_ymax.values[-1]
    obj_ymax = tf.strings.split([obj_ymax], sep='</ymax>')
    obj_ymax = obj_ymax.values[0]

    # NIKHIL - Note the order, This is as per the tf_example_decoder
    return tf.convert_to_tensor([obj_class, obj_ymin, obj_xmin, obj_ymax, obj_xmax])

  # If number of boxes is more than 0 then get box class and location, otherwise class = 'none'
  # XMLDecoder will take care of handling case of 'none'
  bndboxes = tf.cond(tf.greater(num_objects, 0), 
                      lambda: tf.map_fn(get_box_class_and_loc, image_objects),
                      lambda: tf.convert_to_tensor([[b'none', b'-1', b'-1', b'-1', b'-1']]))

  # Flatten
  bndboxes = tf.reshape(bndboxes, [-1])

  # Returned concatenated tensor of image name and bounding box info
  return tf.concat([[image_name], bndboxes], axis=0)


def xml_read_dataset(input_files, config):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.gfile.Glob(input_files)
  num_readers = config.num_readers

  # TODO: Nikhil - Fixing 10 readers in config file as in our case we have only one 
  # if num_readers > len(filenames):
  #   num_readers = len(filenames)
  #   tf.logging.warning('num_readers has been reduced to %d to match input file '
  #                      'shards.' % num_readers)
  
  # filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  filename_dataset = tf.data.TextLineDataset(filenames)

  if config.shuffle:
    filename_dataset = filename_dataset.shuffle(
        config.filenames_shuffle_buffer_size)
  elif num_readers > 1:
    tf.logging.warning('`shuffle` is false, but the input data stream is '
                       'still slightly shuffled since `num_readers` > 1.')
  filename_dataset = filename_dataset.repeat(config.num_epochs or None)
  
  # TODO: Nikhil - Can we split the dataset into 10 txt files and then use this instead of .map() ?? 
  # records_dataset = filename_dataset.apply(
  #     tf.contrib.data.parallel_interleave(
  #         lambda filename: nikhil_parse_function(filename),
  #         cycle_length=num_readers,
  #         block_length=config.read_block_length,
  #         sloppy=config.shuffle))

  # TODO: Nikhil - Check if using num_parallel_calls argument improves speed.
  # If so, can set num_parallel_calls = num_readers
  records_dataset = filename_dataset.map(xml_parse_function)

  if config.shuffle:
    records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
  return records_dataset

############################
##### NIKHIL End ###########
############################

def build(input_reader_config, batch_size=None, transform_input_data_fn=None):
  """Builds a tf.data.Dataset.

  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    batch_size: Batch size. If batch size is None, no batching is performed.
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.

  Returns:
    A tf.data.Dataset based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file,
        use_display_name=input_reader_config.use_display_name,
        num_additional_channels=input_reader_config.num_additional_channels)

    def process_fn(value):
      """Sets up tf graph that decodes, transforms and pads input data."""
      processed_tensors = decoder.decode(value)
      if transform_input_data_fn is not None:
        processed_tensors = transform_input_data_fn(processed_tensors)
      return processed_tensors

    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], input_reader_config)
    if input_reader_config.sample_1_of_n_examples > 1:
      dataset = dataset.shard(input_reader_config.sample_1_of_n_examples, 0)
    # TODO(rathodv): make batch size a required argument once the old binaries
    # are deleted.
    if batch_size:
      num_parallel_calls = batch_size * input_reader_config.num_parallel_batches
    else:
      num_parallel_calls = input_reader_config.num_parallel_map_calls
    dataset = dataset.map(
        process_fn,
        num_parallel_calls=num_parallel_calls)
    if batch_size:
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)

    # Nikhil Added
    print("Nikhil - Removing errors !!!")
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset

  raise ValueError('Unsupported input_reader_config.')


############################
##### NIKHIL Start #########
############################
def xml_dataset_build(input_reader_config, batch_size=None, transform_input_data_fn=None):
  """Builds a tf.data.Dataset.

  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    batch_size: Batch size. If batch size is None, no batching is performed.
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.

  Returns:
    A tf.data.Dataset based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    
    decoder = tf_example_decoder.XMLDecoder(label_map_proto_file=label_map_proto_file)

    def process_fn(value):
      """Sets up tf graph that decodes, transforms and pads input data."""
      processed_tensors = decoder.decode(value)
      if transform_input_data_fn is not None:
        processed_tensors = transform_input_data_fn(processed_tensors)
      return processed_tensors

    dataset = xml_read_dataset(config.input_path[:], input_reader_config)
    
    if input_reader_config.sample_1_of_n_examples > 1:
      dataset = dataset.shard(input_reader_config.sample_1_of_n_examples, 0)
   
    # TODO(rathodv): make batch size a required argument once the old binaries
    # are deleted.
    if batch_size:
      num_parallel_calls = batch_size * input_reader_config.num_parallel_batches
    else:
      num_parallel_calls = input_reader_config.num_parallel_map_calls

    dataset = dataset.map(
        process_fn,
        num_parallel_calls=num_parallel_calls)

    if batch_size:
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)

    # Nikhil Added
    print("Nikhil - Removing errors !!!")
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    
    return dataset

  raise ValueError('Unsupported input_reader_config.')

############################
##### NIKHIL End ###########
############################