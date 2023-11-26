import tensorflow as tf
import sys

# https://www.tensorflow.org/lite/guide/ops_select

converter = tf.lite.TFLiteConverter.from_saved_model(sys.argv[1])
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

open(sys.argv[2] + ".tflite", "wb").write(tflite_model)