from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

train = MNIST(root="./data/", train=True, download=True, transform=ToTensor())
test  = MNIST(root="./data/", train=False, download=True, transform=ToTensor())
# print([train[i][1] for i in range(10)])
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(train[i][0].reshape(28, 28), 'gray')
# plt.show()

# X = train[0][0].reshape(784)

X = np.array([np.array(d[0]).reshape(784) for d in train])
Y = to_categorical(np.array([d[1] for d in train]))
X_test = np.array([np.array(d[0]).reshape(784) for d in test])
Y_test = to_categorical(np.array([d[1] for d in test]))

model = Sequential()
model.add(Dense(256, activation='sigmoid', input_shape=(784,)))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(rate=0.1))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['acc'])
history = model.fit(x=X, y=Y, batch_size=500, epochs=5, validation_split=0.2)

# print(history.history)

# plt.plot(history.history['acc'], label='acc')
# plt.plot(history.history['val_acc'], label='val_acc')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='best')
# plt.show()

test_loss, test_acc = model.evaluate(x=X_test, y=Y_test)
print(test_loss)
print(test_acc)

pred = model.predict(X[100:110])
print(np.argmax(pred, axis=1))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X[100+i].reshape(28, 28), 'gray')
plt.show()