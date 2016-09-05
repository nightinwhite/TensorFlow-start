#incoding:utf-8
import tensorflow as tf
import numpy as np
a = tf.constant(np.full((10),2))
o = tf.nn.dropout(a, 0.5)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
print sess.run(o)
sess.close()