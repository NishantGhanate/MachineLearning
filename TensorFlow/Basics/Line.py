import tensorflow as tf


sess = tf.Session()

const_1 =  tf.constant(value = [2.0])
const_2 =  tf.constant(value = [4.0])

results  = tf.add(x=const_1 , y = const_2 , name = 'results')
# y = Mx + b

M = tf.constant(value = [2.0])
b = tf.constant(value = [5.0])
x = tf.placeholder(dtype=tf.float32)

mult = tf.multiply(x=M , y = x)
y = tf.add(x = mult , y=b)
print(sess.run( fetches = y , feed_dict= {x : [2.0 , 5.0 , 80.0]}))