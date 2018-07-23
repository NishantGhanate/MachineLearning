import tensorflow as tf

const_1 = tf.constant(  value = [[1.0,2.0]],
                        dtype = tf.float32,
                        shape = (1,2),
                        name ='const_1',
                        verify_shape = True
                    )

const_2 = tf.constant(  value = [[3.0 , 4.0]],
                        dtype = tf.float32,
                        shape = (1,2),
                        name = 'const_2',
                        verify_shape = True
                    )

session = tf.Session()
print(  session.run(fetches=[const_1,const_2])  )
