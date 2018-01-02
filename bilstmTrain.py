import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

import pos_data_utils

BATCH_SIZE = 1000
WORKERS = 2
EPOCH = 100
HIDDEN_SIZE = 128
LR = 0.01
NUM_LAYERS = 1
CONTEXT_SIZE = 5
PRETRAIN_EMBEDDINGS = True
USE_SUBWORDS = False

vocab = pos_data_utils.read_vocab(pos_data_utils.vocab_path)
embeddings = np.random.randn(len(vocab), 50) / np.sqrt(len(vocab))
EMBEDDING_DIM = len(embeddings[0])
INPUT_SIZE = EMBEDDING_DIM


vocab_reversed = {}
if PRETRAIN_EMBEDDINGS:
    embeddings = pos_data_utils.read_embeddings(pos_data_utils.wv_path)
if USE_SUBWORDS:
    embeddings, vocab, vocab_reversed = pos_data_utils.generate_embeddings_with_prefixes(embeddings, vocab, EMBEDDING_DIM)


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_embeddings):
        super(RNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(HIDDEN_SIZE*2, len(pos_data_utils.POS_TAGS))


    def forward(self, x):
        embeds = self.embeddings(x)

        (h0, c0) = (Variable(torch.zeros(NUM_LAYERS * 2, BATCH_SIZE, HIDDEN_SIZE)),
         Variable(torch.zeros(NUM_LAYERS * 2, BATCH_SIZE, HIDDEN_SIZE)))

        out, self.hidden = self.lstm(embeds, (h0, c0))

        out = self.linear(out[:, -1, :])

        out = F.log_softmax(out)

        # out = self.out(output[:, -1, :])
        return out


train_data_loader = pos_data_utils.prepare_tensor_dataset(pos_data_utils.pos_train, vocab, WORKERS, BATCH_SIZE)
dev_data_loader = pos_data_utils.prepare_tensor_dataset(pos_data_utils.pos_dev, vocab, WORKERS, BATCH_SIZE)
test_data_loader = pos_data_utils.prepare_tensor_dataset(pos_data_utils.pos_test, vocab, WORKERS, BATCH_SIZE, include_y=False)

model = RNN(len(vocab), EMBEDDING_DIM, embeddings)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

dev_losses = []
train_losses = []
acceptances = []
iterations = []
print("Starting training loop")
for idx in range(0, EPOCH):
    for iteration, batch in enumerate(train_data_loader, 1):
        x, y = Variable(batch[0]), Variable(batch[1])
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print ('Epoch [%d/%d], Loss: %.4f'
           % (idx + 1, EPOCH, loss.data[0]))


    # if idx % 1 == 0:
    #     # calculate accuracy on validation set
    #     dev_loss = 0
    #     model.eval()
    #     correct = 0.0
    #     total = 0.0
    #     for dev_batch_idx, dev_batch in enumerate(dev_data_loader):
    #         x, y = Variable(dev_batch[0]), Variable(dev_batch[1])
    #         output = model(x)
    #         dev_loss = criterion(output, y)
    #         _, predicted = torch.max(output.data, 1)
    #         total += dev_batch[1].size(0)
    #         correct += (predicted == dev_batch[1]).sum()
    #
    #     acc = correct / total
    #
    #     acceptances.append(acc)
    #     train_losses.append(loss.data[0])
    #     dev_losses.append(dev_loss.data[0])
    #     iterations.append(idx)
    #     print("Epoch {: >8}     TRAIN_LOSS: {: >8}      DEV_LOSS: {: >8}     ACC: {}".format(idx, loss.data[0], dev_loss.data[0], acc))
