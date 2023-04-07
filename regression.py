from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

concrete_data = pd.read_csv("./dataset/concrete_data.csv")

# split data into predictor and target
predictors = concrete_data.drop(columns=['Strength'])
target = concrete_data['Strength']

# normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
# save number of predictors
n_cols = predictors_norm.shape[1]


def regression_model():
    # model with two hidden layers, each of 50 hidden units
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


my_model = regression_model()
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=4)
# a value of 2 prints a one-line summary after each epoch.
my_model.fit(X_train, y_train, epochs=100, verbose=2)
loss_metrics = my_model.evaluate(X_test, y_test)
predicted = my_model.predict(X_test)

if __name__ == '__main__':
    print(mean_squared_error(y_test, predicted))
    print(mean_absolute_error(y_test, predicted))
    print(r2_score(y_test, predicted))
