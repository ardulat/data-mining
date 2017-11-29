import matplotlib
matplotlib.use('Agg')
import random
import math
import pandas as pd
import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import tensorflow as tf
#config = tf.ConfigProto()

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

def get_data(file_name, batch):
    data = pd.read_csv(file_name, header=None)
 #   zero_data = pd.DataFrame(np.zeros(shape=(data.shape[0], 80)))
 #   data = pd.concat([data, zero_data], axis=1, ignore_index=True)

    train = data.loc[0:1450, :]
    test = data.loc[1451:, :]

    # Preprocessing/scaling using min-max
    train_X = X_scaler.fit_transform(train.loc[:, 1:batch])
    train_Y = Y_scaler.fit_transform(train.loc[:, (batch + 1):])

    test_X = X_scaler.transform(test.loc[:, 1:batch])
    test_Y = Y_scaler.transform(test.loc[:, (batch + 1):])

    train_X = np.expand_dims(train_X.T, axis=2)
    train_Y = np.expand_dims(train_Y.T, axis=2)

    test_X = np.expand_dims(test_X.T, axis=2)
    test_Y = np.expand_dims(test_Y.T, axis=2)

    return train_X, train_Y, test_X, test_Y


train_X, train_Y, test_X, test_Y = get_data('dm-final-train.txt', 100)


X_length = train_X.shape[0]
Y_length = train_Y.shape[0]


output_dim = input_dim = train_X.shape[-1] 
hidden_dim = 50
layers_stacked_count = 2  

# Optmizer:
learning_rate = 0.0001 
nb_iters = 10000
lr_decay = 0.92 
momentum = 0.5  
lambda_l2_reg = 0.0001


try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()
# sess.close()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):

    encoder = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(X_length)
    ]

    labels = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="labels".format(t))
        for t in range(Y_length)
    ]

    zeros_list = [tf.zeros([X_length - Y_length, 1], "float")]

    decoder = [tf.zeros_like(encoder[0], dtype=np.float32, name="GO")] + encoder[:-1]

    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # For reshaping the input and output dimensions of the seq2seq RNN:
    w_in = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
    b_in = tf.Variable(tf.random_normal([hidden_dim], mean=1.0))
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))

    reshaped_inputs = [tf.nn.relu(tf.matmul(i, w_in) + b_in) for i in encoder]

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    decoder_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        encoder,
        decoder,
        cell
    )

    scale = tf.Variable(1.0, name="SF")
    reshaped_outputs = [scale * (tf.matmul(i, w_out) + b_out) for i in decoder_outputs]

# Training loss and optimizer
with tf.variable_scope('Loss'):
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, labels):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


def train_batch():
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """

    X, Y = train_X, train_Y
    feed_dict = {encoder[t]: X[t] for t in range(len(encoder))}
    feed_dict.update({labels[t]: Y[t] for t in range(len(labels))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch():
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """
    
    X, Y = test_X, test_Y
    feed_dict = {encoder[t]: X[t] for t in range(len(encoder))}
    feed_dict.update({labels[t]: Y[t] for t in range(len(labels))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(nb_iters + 1):
    train_loss = train_batch()
    train_losses.append(train_loss)

    if t % 100 == 0:
        # Tester
        test_loss = test_batch()
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, nb_iters, train_loss, test_loss))

print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))


# Plot loss over time:
plt.figure(figsize=(12, 6))
plt.plot(
    np.array(range(0, len(test_losses)))/float(len(test_losses)-1)*(len(train_losses)-1), 
    np.log(test_losses), 
    label="Test loss"
)
plt.plot(
    np.log(train_losses), 
    label="Train loss"
)
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
out_png = 'losses.png'
plt.savefig(out_png, dpi=150)
plt.show(block=True)

# Test
nb_predictions = 5
print("Let's visualize {} predictions with our signals:".format(nb_predictions))

# X, Y = generate_x_y_data(isTrain=False, batch_size=nb_predictions)
#X, Y = test_X, test_Y
data = pd.read_csv('dm-final-testdist.txt', header = None)
X = data.loc[:, 1:]
X = X_scaler.transform(X)
X = np.expand_dims(X.T, axis=2)

feed_dict = {encoder[t]: X[t] for t in range(X_length)}
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

results = np.squeeze(outputs, axis=2)
results = results.T
results = results[:,0:20]

# rescale
results = Y_scaler.inverse_transform(results)

df = pd.DataFrame(data=results.astype(float))
df.to_csv('predictions.txt', sep=' ', header=False, float_format='%.2f', index=False)

for j in range(nb_predictions):
    plt.figure(figsize=(12, 2))

    for k in range(output_dim):
        past = X[:, j, k]
        #expected = Y[:, j, k]
        pred = outputs[:, j, k]

        label1 = "Seen (past) values" if k == 0 else "_nolegend_"
        #label2 = "True future values" if k == 0 else "_nolegend_"
        label3 = "Predictions" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        #plt.plot(range(len(past), len(expected) + len(past)), expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    out_png = 'out_file.png'
    plt.savefig(out_png, dpi=150)
    plt.show(block=True)

print("Reminder: the signal can contain many dimensions at once.")
print("In that case, signals have the same color.")
print("In reality, we could imagine multiple stock market symbols evolving,")
print("tied in time together and seen at once by the neural network.")