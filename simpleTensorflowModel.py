import tensorflow as tf

def forward_pass(w, x):
    '''takes two parameters w and x
       returns multiplication of w and x'''
    return tf.matmul(w, x)

def train_loop(x, niter = 5):
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        w = tf.get_variable("weights",
         shape = (1,2),
         initializer = tf.truncated_normal_initializer(),
         trainable = True)   

        '''try tio initialize a variable in tensorflow in that way.
           w is initialized here, we are not running it as we know tensorflow follows
           lazy evaluation.So I tried to initialize it using tf's truncated initilizer.
           This a very common initializer in tensorflow neural netowork programs.
           And lastly the variable is trainable so we flagged the variable as True
        '''

        preds =[]
        for i in range(niter):
            preds.append(forward_pass(w,x))
            w = w + 0.01 #gradient update
    return preds

with tf.Session() as sess:
    preds = train_loop(tf.constant([[3.2, 7.8, 9,8],[2.6, 8.2, 1,2]]))
    tf.global_variables_initializer().run()
    for i in range(len(preds)):
        print('{} :{}'.format(i,preds[i].eval()))

'''result is :
    0 :[[-0.8505767 -2.0587177 -2.4416046 -2.1616273]]
    1 :[[-0.79257673 -1.8987179  -2.3416047  -2.0616274 ]]
    2 :[[-0.7345767 -1.7387177 -2.2416046 -1.9616272]]
    3 :[[-0.6765767 -1.5787177 -2.1416044 -1.8616273]]
    4 :[[-0.61857677 -1.418718   -2.0416045  -1.7616274 ]]
'''

