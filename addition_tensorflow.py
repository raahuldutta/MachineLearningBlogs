import tensorflow as tf
''' importing tensrflow linrary as tf, 
    if you havn't installed tensorflow till now,
    please go to terminal and use the command ;
    pip install tensorflow
    I am using now 1.12, most stable version till'''

a = tf.constant([1, 2, 3])  #initialize the first array
b = tf.constant([ 4, 5, 6]) #initialize the second array

c = tf.add(a,b) 
#print(c)
''' the array addition process will not be computed here,
    tensorflow follws the lazy evaluation process so, 
    only the DAG will be built.if you print c here using print(c)
    the output will be : Tensor("Add:0", shape=(3,), dtype=int32)
    '''

with tf.Session() as sess:
    result = sess.run(c)
    print(result)

''' we are excuting the graph under this with section, 
    the graph will be computed, and we will get the result of c
    the output will be : [5 7 9]
    '''
