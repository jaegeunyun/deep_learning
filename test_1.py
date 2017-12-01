import tensorflow as tf
import numpy as np
import time
import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets
start_time = time.time()

boston = datasets.load_boston()
boston_slice = [x[5] for x in boston.data]
data_x = np.array(boston_slice).reshape(-1,1)
data_y = boston.target.reshape(-1,1)
print(data_x.shape, data_y.shape)

n_sample = data_x.shape[0]
X = tf.placeholder(tf.float32, shape=(n_sample,1), name='X')
y = tf.placeholder(tf.float32, shape=(n_sample,1), name='y')
W = tf.Variable(tf.zeros((1,1)), name='weights')
b = tf.Variable(tf.zeros((1,1)), name='bias')

y_pred = tf.matmul(X,W) + b
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)	#AdagradOptimizer, AdamOptimizer
train_op = optimizer.minimize(loss)
summary_op = tf.summary.scalar('loss', loss)

def plot_graph(y, fout):
  '''data plot function. input(feature value), output(cost)'''
  plt.scatter(data_x.reshape(1,-1)[0], boston.target.reshape(1,-1)[0])
  plt.plot(data_x.reshape(1,-1)[0], y.reshape(1,-1)[0])
  plt.savefig(fout)
  plt.clf()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
  y_pred_before = sess.run(y_pred, {X: data_x})
  plot_graph(y_pred_before, 'before.png')
  for i in range(100):
    loss_t, summary, _ = sess.run([loss, summary_op, train_op], feed_dict={X: data_x, y: data_y})
    summary_writer.add_summary(summary,i)
    if i%10 == 0:
      print('loss = % 4.4f' % loss_t.mean())
      y_pred_after = sess.run(y_pred, {X: data_x})
  y_pred_after = sess.run(y_pred, {X: data_x})
  plot_graph(y_pred_after, 'after.png')

end_time = time.time()
print('time spent:', end_time - start_time)

