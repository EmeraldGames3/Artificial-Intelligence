import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

# Load the training data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

# Split into labels and features
y_train = train_data['label']
X_train = train_data.drop(columns=['label'])

# Normalize the features to be between 0 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(test_data)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# Build the model

# model = tf.keras.models.Sequential([
#     layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)),
#     layers.Dropout(0.3),
#     layers.BatchNormalization(),
#     layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ])

model = tf.keras.models.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# model = tf.keras.models.Sequential([
#     layers.Flatten(input_shape=(X_train.shape[1],)),
#     layers.Dense(256, activation='relu'),   # ReLU for general hidden layer
#     layers.Dense(128, activation='tanh'),   # Tanh can handle negative values well
#     layers.Dense(64, activation='relu'),    # ReLU again for consistency and efficiency
#     layers.Dense(10, activation='softmax')  # Softmax for output probability distribution
# ])

# # Define the sequential model framework
# model = tf.keras.models.Sequential([
#     # First dense layer with 512 neurons, using ReLU activation for non-linearity
#     # This large layer is intended to capture a broad range of features from the input
#     layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
#     # Batch normalization to ensure that the model's input layer has mean output close to 0 and standard deviation close to 1
#     # This helps in stabilizing the learning process
#     layers.BatchNormalization(),
#     # Dropout layer to prevent overfitting by randomly setting 30% of the input units to 0 at each update during training
#     layers.Dropout(0.3),
#
#     # Second dense layer with 256 neurons, continuing to refine feature extraction
#     layers.Dense(256, activation='relu'),
#     # Batch normalization layer to maintain normalized output
#     layers.BatchNormalization(),
#     # Dropout layer to reduce overfitting
#     layers.Dropout(0.3),
#
#     # Third dense layer with 128 neurons, focusing on a more refined level of feature extraction
#     layers.Dense(128, activation='relu'),
#     # Batch normalization layer to ensure stability in outputs
#     layers.BatchNormalization(),
#     # Dropout layer to reduce overfitting
#     layers.Dropout(0.2),
#
#     # Fourth dense layer with 64 neurons, used to finalize the extraction of complex relationships in the data
#     layers.Dense(64, activation='relu'),
#     # Batch normalization to normalize outputs
#     layers.BatchNormalization(),
#     # Dropout layer to minimize overfitting at this deeper level
#     layers.Dropout(0.2),
#
#     # Output layer with 10 neurons, one for each digit (0-9)
#     # Softmax activation is used to convert logits to probabilities which sum to 1
#     layers.Dense(10, activation='softmax')
# ])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))


# Evaluate the model
predictions = model.predict(X_test)


# Prepare output
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': np.argmax(predictions, axis=1)
})
submission.to_csv('submission.csv', index=False)


# Plot the data
import matplotlib.pyplot as plt

# Extract accuracy history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

# Plotting the training and validation accuracy
plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
