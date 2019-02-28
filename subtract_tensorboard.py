import tensorflow as tf

a = tf.add(1, 2, name = "Addition1")
b = tf.multiply(a, 3, name = "Multiplication1")
c  = tf.add(4, 5, name = "Addition2")
d = tf.multiply(c, 7, name = "Multiplication2")
f = tf.div(c, a, name = "Division1")


with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(f))
    writer.close()







 

