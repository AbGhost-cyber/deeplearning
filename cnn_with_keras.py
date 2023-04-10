
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape [samples,row, column, channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize data
X_train = X_train / 255
X_test = X_test / 255

# convert target variable to binary
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-10
num_classes = y_train.shape[1]


# convolutional layer with one set of convolutional and pooling layers
def convolutional_model(use_two=False):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if use_two:
        model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


my_model = convolutional_model(use_two=True)
my_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
# The evaluate function returns a list of evaluation metrics (e.g., loss and accuracy)
scores = my_model.evaluate(X_test, y_test, verbose=0)
print(scores)
print("Accuracy: {} \n Error: {}".format(scores[1], 100 - scores[1] * 100))

if __name__ == '__main__':
    print()
