import tensorflow as tf
hello = tf.constant('Hello, TensorFlow new !')
sess = tf.Session()
print(sess.run(hello))
