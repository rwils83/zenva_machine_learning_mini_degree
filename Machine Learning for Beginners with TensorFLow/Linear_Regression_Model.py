import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

m = 2
b = 0.5
x = np.linspace(0, 4, 100)
y = m * x + b + np.random.randn(*x.shape) + .25
plt.scatter(x, y)

class Model:
    def __init__(self):
        self.weight = tf.Variable(10.0)
        self.bias = tf.Variable(10.0)
        # Part 3 Below

    def __call__(self, x):
        return self.weight * x + self.bias
def calculate_loss(y_actual, y_output):
    return tf.reduce_mean(tf.square(y_actual - y_output))
def train(model, x, y, learning_rate):
    # Higher learning_rate faster, but less fine tuning
    # Lower learning_rate slower, more fine tuning
    with tf.GradientTape() as gt:
        y_output = model(x)
        loss = calculate_loss(y, y_output)

    new_weight, new_bias = gt.gradient(loss, [model.weight, model.bias])
    model.weight.assign_sub(new_weight * learning_rate)
    model.bias.assign_sub(new_bias * learning_rate)

model = Model()
epochs = 100
learning_rate = 0.15
for epoch in range(epochs):
    y_output = model(x)
    loss = calculate_loss(y, y_output)
    train(model, x, y, learning_rate)
    template = "Epoch: {}, Loss: {}"
    print(template.format(epoch, loss.numpy()))
print(model.weight.numpy())
print(model.bias.numpy())
new_x = np.linspace(0, 4, 100)
new_y = model.weight.numpy() * new_x * model.bias.numpy()

plt.imshow(plt.scatter(new_x, new_y), interpolation='nearest', cmap='gray')

plt.scatter(x, y)
