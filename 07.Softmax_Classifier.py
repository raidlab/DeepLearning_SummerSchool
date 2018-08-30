import tensorflow as tf

# x_data / y_data 만들기
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Placeholder X, Y 정의
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

# W, b값 초기화
W = tf.Variable(tf.random_uniform([4, 3]), name='weight')
b = tf.Variable(tf.random_uniform([3]), name='bias')

# Hypothesis 정의 (Softmax 함수)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cost Function 정의
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# Gradient Descent 기반 Minimizing Cost 구현
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# 모든 변수 초기화
init = tf.global_variables_initializer()

# 세션 정의
sess = tf.Session()
sess.run(init)

# 계산 (Step, Cost 값 출력)
for step in range(10001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

print('--------------')

# 테스트: One-hot encoding

Test = sess.run(hypothesis, feed_dict={X: [[1, 2, 1, 1], [4, 1, 5, 5], [1, 6, 6, 6]]})
print(Test, sess.run(tf.argmax(Test, 1)))
