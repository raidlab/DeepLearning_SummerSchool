import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

sess = tf.Session()
print(sess.run(adder_node,feed_dict={a:3, b:4.5}))
print(sess.run(adder_node,feed_dict={a:[1,3], b:[2,4]}))