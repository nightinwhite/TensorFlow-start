#incoding:utf-8
import tensorflow as tf
import numpy as np
import Get_Date
print Get_Date.SparseDateFrom([[1,2,3,4,5],[2,3,4]])
# batchsize = 10
# a = tf.placeholder("float", [None], "images")
# b = tf.constant(0.)
# for i in range(int(a.get_shape()[0])):
#     b += a[i]
# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()
# sess.run(init)
# print sess.run(b, feed_dict={a: np.arange(2, 12)})
# sess.close()