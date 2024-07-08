# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the model
model = Sequential([
	Flatten(input_shape=(28, 28)),
	Dense(128, activation='relu'),
	Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Plot the first 10 test images and their predicted labels
predictions = model.predict(test_images[:10])
predicted_labels = np.argmax(predictions, axis=1)

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.title(f'Predicted: {predicted_labels[i]}')
	plt.axis('off')
plt.show()

model.save('mnist_model')
