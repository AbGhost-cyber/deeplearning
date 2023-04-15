from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers import Dense
from keras import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np


def regression_model(n_cols):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


concrete_data = pd.read_csv('./dataset/concrete_data.csv')

features = concrete_data.drop(columns=['Strength'])
label = concrete_data['Strength']

# X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=4)
#
# r_model = regression_model(n_cols=features.shape[1])
# r_model.fit(X_train, y_train, epochs=50, verbose=2)
# y_hat = r_model.predict(X_test)
# scores = r_model.evaluate(X_test, y_test, verbose=0)
mean_squared_errors = []
# for i in range(50):
#     X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=4)
#
#     r_model = regression_model(n_cols=features.shape[1])
#     r_model.fit(X_train, y_train, epochs=50, verbose=2)
#     y_hat = r_model.predict(X_test)
#     scores = r_model.evaluate(X_test, y_test, verbose=0)
#     mse = mean_squared_error(y_test, y_hat)
#     mean_squared_errors.append(mse)

# using normalization
features_norm = (features - features.mean()) / features.std()

for i in range(50):
    print(f"step: {i + 1}")
    X_train, X_test, y_train, y_test = train_test_split(features_norm, label, test_size=0.3, random_state=4)

    r_model = regression_model(n_cols=features_norm.shape[1])
    r_model.fit(X_train, y_train, epochs=100, verbose=2)
    y_hat = r_model.predict(X_test)
    scores = r_model.evaluate(X_test, y_test, verbose=0)
    mse = mean_squared_error(y_test, y_hat)
    mean_squared_errors.append(mse)

if __name__ == '__main__':
    print(np.mean(mean_squared_errors))
    print(np.std(mean_squared_errors))
