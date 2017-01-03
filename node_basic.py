import tensorflow as tf

sess = tf.InteractiveSession()

# n x 50 x 50 input
images = tf.placeholder(tf.bool, shape=[None, 2500])
# n x 1 output
labels = tf.placeholder(tf.bool, shape=[None, 1])

# weights and bias init 0
W = tf.Variable(tf.zeros([2500,1]))
b = tf.Variable(tf.zeros([1]))

# init variables
sess.run(tf.global_variables_initializer())

# set predictions
predictions = tf.sigmoid(tf.matmul(images, W) + b)