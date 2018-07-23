import tensorflow as tf
session = tf.Session()

placeholder_1 = tf.placeholder( dtype = tf.float32,
                                shape = [1,4],
                                name = 'placeholder_1'
                                )

placeholder_2 = tf.placeholder( dtype = tf.float32,
                                shape = [2,2],
                                name = 'placeholder_2'
                                )

print(placeholder_1)

print(session.run(fetches = [placeholder_1 , placeholder_2],
                    feed_dict= {placeholder_1 : [[95.0,64.0,10,71.0]] ,placeholder_2 : [[21.0 , 13.0] , [3.0 , 4.0] ]}))