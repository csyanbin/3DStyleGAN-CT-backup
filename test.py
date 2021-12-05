import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

