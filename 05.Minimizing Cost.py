import tensorflow as tf
import matplotlib.pyplot as plt

# x_data / y_data 만들기
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# x_data 크기
m = len(x_data)

# Placeholder W 정의
W = tf.placeholder(tf.float32)

# Hypothesis 정의
hypothesis = W * x_data

# Cost Function 정의
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

# 모든 변수 초기화 및 세션 정의
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# W , cost 값 저장 할 리스트 초기화
W_val, cost_val = [], []

# 계산(W 값 변화에 따른 Cost 값)
for i in range(-30, 51):
    xPos = i*0.1                                    # W: x 좌표 -3에서 5까지 0.1씩 증가
    yPos = sess.run(cost, feed_dict={W: xPos})      # Cost: y 좌표

    # W, cost 값을 리스트에 저장
    W_val.append(xPos)
    cost_val.append(yPos)

sess.close()

# W, cost 값 그래프 그리기
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()