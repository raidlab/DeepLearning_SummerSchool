import tensorflow as tf

# x_data / y_data 만들기
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# Placeholder X, Y 정의
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W, b값 초기화
W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Hypothesis 정의 (Sigmoid 함수)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Cost Function 정의
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Gradient Descent 기반 Minimizing Cost 구현
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# Logistic Classifier 및 정확도 측정
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

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

# 결과 확인 (Hypothesis, Correct (Y), Accuracy)
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)