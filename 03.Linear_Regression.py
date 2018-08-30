import tensorflow as tf

# x_data / y_data 만들기
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# W, b값 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Hypothesis 정의
hypothesis = W * x_data + b

# Cost Function 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Gradient Descent 기반 Minimizing Cost 구현
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# 모든 변수 초기화
init = tf.global_variables_initializer()

# 세션 정의
sess = tf.Session()
sess.run(init)

# 계산 (Step, Cost, W,b 값 출력)
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
