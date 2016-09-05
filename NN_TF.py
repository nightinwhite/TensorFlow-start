#incoding:utf-8
import tensorflow as tf
import numpy as np
import cv2
#载入数据
# imags_path = "/home/liuyi/test/images"
# ans_name = "answer"
images_path = "0_ABOUT.jpg"
ans_name = "answer"
test_image = cv2.imread(images_path, 0)
test_image = cv2.resize(test_image, (200, 60),interpolation=cv2.INTER_CUBIC)
test_image = test_image.transpose()
test_image = np.resize(test_image, (1, 200, 60, 1))/255.0
#print test_image.shape
# cv2.imshow("1", test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#构建模型
#----定义层----
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, (1, strides, strides, 1), "SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def dropout(x, rate):
    return tf.nn.dropout(x, rate)

def maxpool2d(x, poolsize=(2,2)):
    px = poolsize[0]
    py = poolsize[1]
    return tf.nn.max_pool(x, ksize=(1, px, py, 1), strides=(1, px, py, 1),padding="SAME")

def flatten(x):
    return tf.contrib.layers.flatten(x)

def full_con(x, out_dim):
    return
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wf1': tf.Variable(tf.random_normal([5, 5, 32, 64]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
}
x = tf.placeholder("float", [None, 200, 60, 1], "images")
conv2do1 = conv2d(x, weights['wc1'], biases['bc1'])
conv2do2 = conv2d(conv2do1, weights['wc2'], biases['bc2'])
conv2do2 = maxpool2d(conv2do2)
fc1 = flatten(x)
o = fc1
#运转模型
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
out_images = sess.run(o, feed_dict={x: test_image})
sess.close()
print out_images.shape
# cv2.imshow("1", out_images[0, :, :, 0:3])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
