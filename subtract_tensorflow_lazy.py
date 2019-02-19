import tensorflow as tf

x = tf.constant([69, 34, 98])
y = tf.constant([98, 67, 89])
z1 = x - y  #Common arethmetic operation can be used directly
z2 = x * y
z3 = tf.add(z1, z2)

with tf.Session() as sess:
    print("z1 is {}".format(z1.eval()))
    print("z2 is  {}".format(z2.eval()))
    print("z3 is {}".format(z3.eval()))

''' we are executing the graph using lazy loading, the result is :
    z1 is [-29 -33   9]
    z2 is  [6762 2278 8722]
    z3 is [6733 2245 8731]
'''
    

