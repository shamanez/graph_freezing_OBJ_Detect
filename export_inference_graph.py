

"""
Step 1 - Download all the model-number ckpt files (There will be 3 files ) also downlaod the checkpoint file from the bucket/train 

Step 2 - Paste this file inside models-master/object_detection 

step 3 - Since to keep up with the orientation make a  new foder inside models-master/object_detection  named object_detection and copy the content of original object detection to it . This is necessory to run locally ! 

step 4- Make sure you put the correct number of your checkpoint  in the 3rd flag in this script 

step 5- Then run that from the models/object detection folder 

step 6 - Then you will get an output a .pb file named "output_inference_graph.pb" then rename it as "frozen_inference_graph.pb"

step 7- Put in inside the "ssd_mobilenet_v1_coco_11_06_2017" folder deleting content 

step 8- Put your validation images to the test image 

step 9-Run the object_detection_tutorial.ipynb
 

"""


import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('pipeline_config_path', 'object_detection/samples/configs/faster_rcnn_resnet101_pets.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('checkpoint_path', 'model.ckpt-9286', 'Optional path to checkpoint file. '
                    'If provided, bakes the weights from the checkpoint into '
                    'the graph.')
flags.DEFINE_string('inference_graph_path', 'output_inference_graph.pb', 'Path to write the output '
                    'inference graph.')
flags.DEFINE_bool('export_as_saved_model', False, 'Whether the exported graph '
                  'should be saved as a SavedModel')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.pipeline_config_path, 'TrainEvalPipelineConfig missing.'
  assert FLAGS.inference_graph_path, 'Inference graph path missing.'
  assert FLAGS.input_type, 'Input type missing.'
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  exporter.export_inference_graph(FLAGS.input_type, pipeline_config,
                                  FLAGS.checkpoint_path,
                                  FLAGS.inference_graph_path,
                                  FLAGS.export_as_saved_model)


if __name__ == '__main__':
  tf.app.run()
