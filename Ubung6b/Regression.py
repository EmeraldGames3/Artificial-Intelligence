import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

raw = pd.read_csv("Regression_BSD_hour.csv")

all_days = len(raw) // 24
print("Total observations", len(raw))
print("Total number of days", all_days)

days_for_training = int(all_days * 0.7)
hours_for_training = days_for_training * 24

X_train = raw[0:hours_for_training]
X_test = raw[hours_for_training:]

y_train = X_train['cnt']
y_test = X_test['cnt']

print("Observations for training", X_train.shape)
print("Observations for testing", X_test.shape)


def plot_data(X, y, first_day=3 * 7, duration_days=3 * 7):
    s = first_day * 24  # start hour
    e = s + duration_days * 24  # end hour

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    for x, v in X['workingday'][s:e].items():
        if v == 1:
            ax0.axvline(x, lw=3, c='lightgrey')
            ax1.axvline(x, lw=3, c='lightgrey')

    mid_day_indexes = []
    for x, v in X['hr'][s:e].items():
        if v == 0:
            ax0.axvline(x, ls=':', c='grey')
            ax1.axvline(x, ls=':', c='grey')
        if v == 12:
            mid_day_indexes.append(x)

    for c in ['temp', 'hum', 'windspeed', 'weathersit']:
        ax0.plot(X[c][s:e], label=c)

    ax0.legend(loc="upper left")
    ax0.set_ylabel('Input variables')

    ax1.plot(y[s:e], 'r:', label="ground truth")
    ax1.legend(loc="upper left")
    ax1.set_ylabel('Number of Rentals per hour')

    ax1.set_xticks(mid_day_indexes)
    ax1.xaxis.set_ticklabels([X['dteday'][i] for i in mid_day_indexes], rotation=90)

    plt.tight_layout()
    plt.show()


plot_data(X_train, y_train)

# Drop the 'dteday' column
X_train = X_train.drop(columns=['dteday'])
X_test = X_test.drop(columns=['dteday'])

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print(f"Mean Squared Error: {mse}")
