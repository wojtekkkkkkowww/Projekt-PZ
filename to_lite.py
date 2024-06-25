import tensorflow as tf
import sys

# https://www.tensorflow.org/lite/guide/ops_select

PATH = f'models/{sys.argv[1]}.keras'

converter = tf.lite.TFLiteConverter.from_saved_model(PATH)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

open(PATH + ".tflite", "wb").write(tflite_model)