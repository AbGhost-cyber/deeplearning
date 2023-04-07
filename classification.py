from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

# read data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# With conventional neural networks, we cannot feed in the image as input as is.
# So we need to flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784.
num_pixels = X_train.shape[1] * X_train.shape[2]  # find size of one dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')  # flatten test images

# Since pixel values can range from 0 to 255, let's normalize the vectors to be between 0 and 1.
X_train = X_train / 255
X_test = X_test / 255

# for classification, we need to divide our target variable into categories.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def classification_model():
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# build the model
my_model = classification_model()
my_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate
scores = my_model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# Sometimes, you cannot afford to retrain your model everytime you want to use it,
# especially if you are limited on computational resources and training your model can take a long time.
# Therefore, with the Keras library, you can save your model after training. To do that, we use the save method.
my_model.save('classification_model.h5')
# when you are ready to use the model again you can use the load_model function
pretrained_model = load_model('classification_model.h5')

if __name__ == '__main__':
    print(X_train.shape)
