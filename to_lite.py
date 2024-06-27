import tensorflow as tf
import sys

# https://www.tensorflow.org/lite/guide/ops_select

PATH = f'to_lite_data/{sys.argv[1]}'

converter = tf.lite.TFLiteConverter.from_saved_model(PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

open(f'models/{sys.argv[1]}.tflite', "wb").write(tflite_model)