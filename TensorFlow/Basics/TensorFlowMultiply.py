""" /**
 * @author [Nishant Ghanate]
 * @email [nishant7.ng@gmail.com]
 * @create date 2018-05-30 07:05:08
 * @modify date 2018-05-30 07:05:08
 * @desc [description]
*/
 """
# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()