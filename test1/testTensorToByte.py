import numpy as np
import tensorflow as tf
import torch

# 创建一个长度为10的FLOAT32 tensor，元素值递增
tensor = tf.constant(np.arange(10), dtype=tf.float32)

print(tensor)

# 将tensor转换为numpy数组
numpy_array = tensor.numpy()
print(numpy_array)

# 转为字节数组
byte_array = numpy_array.tobytes()
print(byte_array)

