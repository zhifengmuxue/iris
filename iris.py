from sklearn import datasets
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

'''
    in this code, we active the identification of irises
    the datasets has 150 data, you can see it in "dataset_load_test.py"
    The first 120 pieces of data after random classification were included in train set
    and the others were included in test set
'''
# load the data
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

seed = np.random.randint(10000, high=None, size=None, dtype='l')

# random scrambled the data
np.random.seed(seed)  # seed : each time create the same number
np.random.shuffle(x_data)
np.random.seed(seed)  # data x and y need to correspond
np.random.shuffle(y_data)
tf.random.set_seed(seed)

# divide into the train data and the test data
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# change the data type
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# match the feature and label, certain the size of a batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# build the neural networks: 4 feature means 4 input nod , 3 classes means 3 output nod
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# set learn rate and other parameter
lr = 0.1
epoch = 500
train_loss_results = []  # use a list record each loss
test_acc = []  # use a list record acc
loss_all = 0  # record all loss

# train
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))  # Mean square error
        grads = tape.gradient(loss, [w1, b1])

        # update the parameter
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

        # accumulate the loss
        loss_all += loss

    # calculate average loss for the epoch
    avg_loss = loss_all / (step + 1)

    # print loss for each epoch
    print("Epoch {}, loss: {:}".format(epoch, avg_loss))
    train_loss_results.append(avg_loss)
    loss_all = 0

    # test part
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # use the parameter already update
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct/total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("-----------------------------------")

# draw the loss
plt.title('Loss Functino Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

# draw the Accuracy
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()



