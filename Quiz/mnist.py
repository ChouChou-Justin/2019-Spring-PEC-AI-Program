#%matplotlib inline
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#測試 matshow功能，將矩陣像素資料顯示出其圖形
def samplemat():
    dims = (15, 15)
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    # Display matrix
    # plot a matrix or an array as an image
    plt.matshow(aa)  
    plt.show()

def matPltShow():
    normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
    uniform_samples = np.random.uniform(size = 100000) # 生成 100000 組介於 0 與 1 之間均勻分配隨機變數
    plt.hist(normal_samples)
    plt.show()
    plt.hist(uniform_samples)
    plt.show()

def showMnist():    
    # 讀入 MNIST
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    x_train = mnist.train.images
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # 檢視結構
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print("---")
    # 來看看 mnist 的型態
    print(type(mnist))
    print(mnist.train.num_examples)
    print(mnist.validation.num_examples)
    print(mnist.test.num_examples)

    print("讓我們看一下 MNIST 訓練還有測試的資料集長得如何")
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels
    print(" train_img 的 type : %s" % (type(train_img)))
    print(" train_img 的 dimension : %s" % (train_img.shape,))
    print(" train_label 的 type : %s" % (type(train_label)))
    print(" train_label 的 dimension : %s" % (train_label.shape,))
    print(" test_img 的 type : %s" % (type(test_img)))
    print(" test_img 的 dimension : %s" % (test_img.shape,))
    print(" test_label 的 type : %s" % (type(test_label)))
    print(" test_label 的 dimension : %s" % (test_label.shape,))

    # 檢視一個觀測值
    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    for i in [0, 1, 2]:
        curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
        curr_label = np.argmax(trainlabel[i, :] ) # Label
        plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.title("" + str(i + 1) + "th Training Data " 
              + "Label is " + str(curr_label))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w) 

def mnistDo():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    w = init_weights([784, 10]) 
    py_x = model(X, w)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # 计算误差
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
    predict_op = tf.argmax(py_x, 1) 

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(100):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))

def main():
    showMnist()
    #matPltShow()
    #samplemat()
    
main()    