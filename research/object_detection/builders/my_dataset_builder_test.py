############################
##### NIKHIL Start #########
############################
"""
Tests for dataset_builder for creating dataset from text file.
Still using the input_reader proto format, and within it tf_record_input_reader.
tf_record_input_read.input_path now specifies the name of txt file that contains 
images to be used in dataset.
"""

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append("C:\\Users\\nikhil\\Desktop\\TF_Model_API\\models\\research\\")

from google.protobuf import text_format

from object_detection.builders import dataset_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


class MyDatasetBuilderTest(tf.test.TestCase):

  def test_build_tf_record_input_reader(self):
    tf_record_path = os.path.join(self.get_temp_dir(), 'tfrecord')
    print("Path===>", tf_record_path)

    input_reader_text_proto = """
      shuffle: false
      num_readers: 1
      tf_record_input_reader {{
        input_path: '{0}'
      }}
      label_map_path: '{1}'
    """.format(r"C:\\Users\\nikhil\\Desktop\\TF_Model_API\\data\\train.txt",
               r"C:\\Users\\nikhil\\Desktop\\TF_Model_API\\data\\label_map.pbtxt")
    input_reader_proto = input_reader_pb2.InputReader()
    text_format.Merge(input_reader_text_proto, input_reader_proto)
    tensor_dict = dataset_builder.make_initializable_iterator(
        dataset_builder.xml_dataset_build(input_reader_proto)).get_next()

    with tf.train.MonitoredSession() as sess:
        for i in range(4):
            output_dict = sess.run(tensor_dict)
            print(output_dict)
            for k,v in output_dict.items():
                print(v.shape, v.dtype)


if __name__ == '__main__':
  tf.test.main()

############################
##### NIKHIL End ###########
############################