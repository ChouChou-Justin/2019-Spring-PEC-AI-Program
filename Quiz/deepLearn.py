#https://www.jiqizhixin.com/articles/2017-10-07-3

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

def test30():
    hello = tf.constant('hello tensorflow!') 
    sess = tf.Session()
    print("hello") 
    print(sess.run(hello))
    
def test31():    
    x = np.array([1, 2, 3, 4, 5, 6])
    print(x)
    print(type(x))
    y = x.astype('float64')
    print(y.dtype)
    print(y)

def test32():    
    x = np.ones([2, 3]) 
    print(x)
    y = x.reshape([3, 2])    
    print(y)
    z=np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16],
                [17,18,19,20]])
    
    print(z[:,::-1])    
    print(z[:,0])
    print(z[:,1])
    print(z[:,2:3])
    #np.ones([2, 3]) 設定 2x3 值均為 1
    # x.reshape([3, 2])將 2x3 改為 3x2    
    #z[:,0] 是取矩陣X的所有行的第0列的元素，
    #z[:,1] 是取所有行的第1列的元素。
    #z[:,m:n:s]即取矩陣X的所有行中的的第m到n-1列資料，含左不含右。
    # s 是 step, 空的就是全取，-1是倒轉
    #z[0,:]就是取矩陣X的第0行的所有元素，X[1,:]取矩陣X的第一行的所有元素。
test32()
def test33():    
    x_data = np.random.rand(50,2)
    y_data = ((x_data[:,1] > 0.5)*( x_data[:,0] > 0.5))
    print(x_data)
    print(y_data)
    
def test07():
    sess = tf.Session()
    d0 = tf.Variable(1, dtype=tf.float32, name='number1')
    d1 = tf.tan(d0)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    B = sess.run(d1)
    print(B)       
    
def test06():
    sess = tf.Session()
    d50 = tf.placeholder(tf.float32, name ="input1")
    d51 = tf.sin(d50)
    A=sess.run(d51, feed_dict={d50: 0.2})
    print(A)

def test05():
    A = tf.constant(10, dtype=tf.int64)
    sess = tf.Session()
    B = sess.run(A)
    print(B)

def test04():
    A = tf.constant('Hello World!')
    with tf.Session() as sess:
        B = sess.run(A)
        print(B)
    
def test00():
    # 用 numpy 亂數產生 100 個點
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    # Find values for W and b that compute y_data = W * x_data + b
    # W should be 0.1 and b 0.3, TensorFlow will compute it
    # Tensorflow 逐步 fitting 權重值
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    #W = tf.Variable(tf.random_normal([1]), name='weight')
    #b = tf.Variable(tf.random_normal([1]), name='bias')
    #X = tf.placeholder(tf.float32, shape=[None])
    #Y = tf.placeholder(tf.float32, shape=[None])
    
    y = W * x_data + b
    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)    
    train = optimizer.minimize(loss)
    # Before starting, initialize the variables.  We will 'run' this first.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Launch the graph. Fit the line.
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))
            plt.plot(x_data, y_data, 'ro', label='Original data')
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
            plt.legend()
            plt.show()

def test02():
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])
    print(W, b, X, Y)
    # Our H (hypothesis) XW+b
    H = X * W + b
    # cost/loss function
    cost = tf.reduce_mean(tf.square(H - Y))
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes all variables in the graph.
    sess.run(tf.global_variables_initializer())
    # Fit the line
    for step in range(201):
       # feed in training data x, y  
       cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
       if step % 20 == 0: print(step, cost_val, W_val, b_val)

def test03():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0, tf.float32)
    node3 = tf.add(node1, node2)
    print("node1:", node1, "node2:", node2)
    print("node3:", node3)
    sess = tf.Session()
    print("sese.run(node1, node2):",sess.run([node1, node2]))
    print("sese.run(node3):",sess.run(node3))

def test08():
    sess = tf.Session()
    x=tf.ones(shape=[3, 2],dtype=tf.float32,name=None) 
    y=tf.zeros([2, 3], tf.int32) 
    print(sess.run(x)) 
    print(sess.run(y))

def test09():
    sess = tf.Session()
    a = tf.constant(2,shape=[2,2])
    b = tf.constant([1,2,3],shape=[6])
    c = tf.constant([1,2,3],shape=[3,2])
    print(sess.run(a)) 
    print(sess.run(b))
    print(sess.run(c))

def test10():
    sess = tf.Session()
    a = tf.random_uniform([5], minval=0, maxval=None, dtype=tf.float32, seed=None,name=None) 
    b = tf.random_normal([5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    print(sess.run(a)) 
    print(sess.run(b))

def test11():
    sess = tf.Session()
    a = tf.fill([2,3],2)
    b = tf.constant(2,shape=[4,4])
    c = tf.constant([1,2,3],shape=[3,2])
    print(sess.run(a)) 
    print(sess.run(b))
    print(sess.run(c))

def test12():
    sess = tf.Session()
    data = [2,3,4]
    x = tf.expand_dims(data, 0)
    print(sess.run(x))
    y = tf.expand_dims(data, 1)
    print(sess.run(y))
    z = tf.expand_dims(data, -1)
    print(sess.run(z))
    
    print(sess.run(tf.shape(data)))
    print(sess.run(tf.shape(x)))
    print(sess.run(tf.shape(y)))
    print(sess.run(tf.shape(z)))

def test13():
    sess = tf.Session()
    x = [1, 4]
    y = [2, 5]
    z = [3, 6]
    a = tf.stack([x, z, y])
    b = tf.stack([z, x, y], axis=1) 
    print(sess.run(a))
    print(sess.run(b))
 
def test14():
    sess = tf.Session()
    t1 = [[1, 2, 3], 
          [4, 5, 6]]
    t2 = [[7, 8, 9], 
          [10, 11, 12]]
    a = tf.concat([t1, t2], 0)
    b = tf.concat([t1, t2], 1)
    print(sess.run(a))
    print(sess.run(b))

def test15():
    sess = tf.Session()
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    a = tf.reshape(t, [3, 4])
    print(sess.run(a))

def test16():
    sess = tf.Session()
    #t = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    t = [[1, 2, 3], 
         [2, 3, 4], 
         [5, 4, 3], 
         [8, 7, 2]]
    a = tf.argmax(input=t, axis=0)
    b = tf.argmax(input=t, axis=1)    
    print(sess.run(a))
    print(sess.run(b))    

def test17():
    sess = tf.Session()
    t1 = tf.constant([[1, 2]])
    t2 = tf.constant([[3],
                      [4]])
    a = tf.matmul(t1, t2)
    b = tf.matmul(t2, t1)
    print(sess.run(a))   
    print(sess.run(b))    
    
def test18():
    sess = tf.Session()
    x = [[1, 2, 3],  
         [4, 5, 6],
         [7, 8, 9]]
    a = tf.reduce_sum(x) 
    b = tf.reduce_sum(x, 0)
    e = tf.reduce_sum(x, 0, keep_dims=True)
    c = tf.reduce_sum(x, 1)
    f = tf.reduce_sum(x, 1, keep_dims=True)
    d = tf.reduce_sum(x,[0, 1])
    print(sess.run(a))   
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(f))
def test19():
    sess = tf.Session()
    x = [[1, 2, 3],  
         [4, 5, 6],
         [7, 8 ,9]]
    a = tf.reduce_mean(x) 
    b = tf.reduce_mean(x, 0)
    c = tf.reduce_mean(x, 1)

    print(sess.run(a))   
    print(sess.run(b))
    print(sess.run(c))
       
def test20():
    sess = tf.Session()
    d0 = tf.Variable(1, dtype=tf.float32, name='number1')
    b = tf.Variable(tf.zeros([1]))    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(d0)) 
    print(sess.run(b))     
    
#def main():
#    test20()
#    test19()
#    test18()
#    test17()
#    test16()
#    test15()
#    test14()
#    test13()
#    test12()
#    test11()
#    test10()
#    test09()
#    test08()
#    test07()
#    test06()
#    test05()
#    test04()
#    test03()
#    test02()
#    test00()
    
    
    
#    test30()
#    test31()
#    test32()
#    test33()
    
#main()