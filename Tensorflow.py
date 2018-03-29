

#莫凡Tensorflow教程5

#训练y=x*0.1+0.3 预测的权重和偏值  也就是0.1和0.3


#
# import tensorflow as tf
# import numpy as np
# #create data
# x_data=np.random.rand(100).astype(np.float32)
# y_data=x_data*0.1+0.3
#
# #create tensorflow structure statrt#
# Weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases=tf.Variable(tf.zeros([1]))    #1是一维的意思
#
# y=Weight*x_data+biases   #预测的权重和偏值  也就是0.1和0.3
# loss=tf.reduce_mean(tf.square(y-y_data))  #最开始与预测的误差会很大
#
# optimizer=tf.train.GradientDescentOptimizer(0.6)  #梯度下降优化器  0.5是学习效率，小于1
# train=optimizer.minimize(loss)
# init=tf.initialize_all_variables()  #初始化所有变量
# #create tensorflow structure statrt#
#
# sess=tf.Session()
# sess.run(init)  #激活整个神经网络，切记!
#
# for step in range(500):
#     sess.run(train)  #训练
#     if step%20==0:
#         print(step,sess.run(Weight),sess.run(biases),sess.run(loss))#要run指向权重和偏值，才能显示
#
#
#
#
# import tensorflow as tf
# #计算两个矩阵的乘
# matrix1=tf.constant([[3,3]])  #一行两列的矩阵
# matrix2=tf.constant([[2],
#                     [2]])
# product=tf.matmul(matrix1,matrix2)    #matrix multipy    np.dot(m1,m2)
#
# # ## Session使用方法1
# # sess=tf.Session()
# # result=sess.run(product)
# # print(result)
#
#
# ###Session使用方法2
# with tf.Session() as sess:  #在with后自动被close  不用sess.close()
#     result2=sess.run(product)
#     print(result2)
