from src.models.neural_net import NeuralNetWork
from src.conf import config
from src.models import optimizers
import numpy as np

nn = NeuralNetWork(input_size=784, hidden_size=100, output_size=10, depth=0, batch_size=60, lr=0.3,
                   optimizer=optimizers.SGD)

# nn = NeuralNetWork(load_nn_name="mnist")
train_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_i.npy"))
train_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_t.npy"))
nn.batch_train(train_value, train_label, epochs=10)
acc = nn.accuracy(train_value, train_label)
print(acc)
# if nn.accuracy(train_value, train_label) > acc:
#     nn.save_nn("mnist")
