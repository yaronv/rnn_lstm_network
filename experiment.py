import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import data_utils
from gen_examples import data_dir, TRAIN_EXAMPLES_NAME, TEST_EXAMPLES_NAME

BATCH_SIZE = 50
WORKERS = 2
INPUT_SIZE = 1
EPOCH = 100
HIDDEN_SIZE = 10
LR = 0.001
NUM_LAYERS = 1


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )


        self.out = nn.Linear(HIDDEN_SIZE, 2)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)),
                Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)))

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        r_out, self.hidden = self.lstm(x, self.hidden)

        output, _ = pad_packed_sequence(r_out, batch_first=True)

        # choose r_out at the last time step
        out = self.out(output[:, -1, :])
        return out


def prepare_padded_data(x_tensor, y_tensor):
    b_x = Variable(x_tensor.view(-1, x_tensor.shape[1], INPUT_SIZE))  # reshape x to (batch, time_step, input_size)
    b_y = Variable(y_tensor)  # batch y

    b_x_np = b_x.data.numpy()
    b_y_np = b_y.data.numpy()

    zipped = zip(b_x_np, b_y_np)
    zipped.sort(key=lambda x: len(x[0][x[0] != 0]), reverse=True)
    b_x_np, b_y_np = zip(*(zipped))
    lengths = [len(x[x != 0]) for x in b_x_np]

    b_x_np = np.asarray(b_x_np)
    b_y_np = np.asarray(b_y_np)

    b_x = Variable((torch.from_numpy(b_x_np)).type(torch.FloatTensor))
    b_y = Variable((torch.from_numpy(b_y_np)).type(torch.LongTensor))

    b_x_padded = pack_padded_sequence(b_x, lengths, batch_first=True)

    return b_x_padded, b_y


train_loader = data_utils.prepare_tensor_dataset(os.path.join(data_dir, TRAIN_EXAMPLES_NAME), WORKERS, BATCH_SIZE)
test_loader  = data_utils.prepare_tensor_dataset(os.path.join(data_dir, TEST_EXAMPLES_NAME), WORKERS, BATCH_SIZE)


model = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


train_loss_ = []
train_acc_ = []

# training and testing
for epoch in range(EPOCH):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0

    for step, (x, y) in enumerate(train_loader):        # gives batch data

        b_x, b_y = prepare_padded_data(x, y)

        optimizer.zero_grad()
        # model.zero_grad()
        model.hidden = model.init_hidden()

        output = model(b_x)                               # rnn output
        loss = criterion(output, b_y)                   # cross entropy loss
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        _, predicted = torch.max(output, 1)
        total_acc += sum(predicted.data == b_y.data)
        total += b_y.size()[0]
        total_loss += loss.data[0]

    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)

    # if epoch % 5 == 0:
    correct = 0
    total_test = 0
    for test_x, test_y in test_loader:
        test_x, test_y = prepare_padded_data(test_x, test_y)
        test_output = model(test_x)                   # (samples, time_step, input_size)
        _, predicted = torch.max(test_output, 1) #1].data.numpy().squeeze()

        total_test += test_y.size(0)
        correct += sum(predicted.data == test_y.data)
    accuracy = correct / float(total_test)



    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.3f' % accuracy, '| train loss new: %.3f' % train_loss_[epoch], '| train acc: %.3f' % train_acc_[epoch])