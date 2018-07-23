import tensorflow as tf

# y = Mx+b

x_train = [1.0,2.0,3.0,4.0]
y_train = [-1.0,-2.0,-3.0,-4.0]

M = tf.Variable(initial_value = [1.0] , dtype=tf.float32)
b = tf.Variable(initial_value = [1.0] , dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
y_input = tf.placeholder(dtype=tf.float32)

mult = tf.multiply(x=M , y = x)
y_output = tf.add(x = mult , y=b)

loss = tf.reduce_sum(input_tensor = tf.square( x = y_output - y_input))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(fetches = loss , feed_dict= { x : x_train , y_input : y_train}) )

for _ in range(1000):
    session.run(fetches = train_step , feed_dict= { x : x_train , y_input : y_train})

print( session.run(fetches = [loss,M,b] , feed_dict= { x : x_train , y_input : y_train}) )

print( session.run(fetches = y_output , feed_dict= { x :[5.0 , 10.0, 15.0]}) )