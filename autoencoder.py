import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import seaborn as sns
# from tensorflow.keras.utils import plot_model
# import pydot
# from pydotplus import graphviz

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and flatten the images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = train_images.reshape((len(train_images), np.prod(train_images.shape[1:])))
test_images = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))

# Define the autoencoder model
encoding_dim = 32  # Size of the encoded representation
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on the MNIST dataset
history = autoencoder.fit(train_images, train_images,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(test_images, test_images))

# Obtain the reconstructed images
reconstructed_images = autoencoder.predict(test_images)

# Compute the reconstruction loss for each image
reconstruction_loss = np.mean(np.square(test_images - reconstructed_images), axis=1)

# Extract accuracy and loss from the training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Choose a few test images to visualize
num_images_to_visualize = 4

# Plot the original and reconstructed images
plt.figure(figsize=(10, 4))
for i in range(num_images_to_visualize):
    # Original image
    plt.subplot(2, num_images_to_visualize, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, num_images_to_visualize, i + num_images_to_visualize + 1)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.show()

# Create a plot to visualize accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Set a threshold for identifying tampered data
threshold = 0.02  # Adjust this threshold based on your needs

# Create a new dataset of non-tampered images
non_tampered_images = test_images[reconstruction_loss < threshold]
non_tampered_labels = np.zeros(len(non_tampered_images))  # Labels indicating non-tampered (0)

# Store non-tampered data as 'X_train', 'y_train', 'X_test', 'y_test'
X_train = np.concatenate((train_images, non_tampered_images), axis=0)
y_train = np.concatenate((train_labels, non_tampered_labels), axis=0)

X_test = test_images
y_test = test_labels

# Save non-tampered data and labels to an HDF5 file
output_file_path = 'non_tampered_dataset.h5'
with h5py.File(output_file_path, 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)

print('Non-tampered dataset saved to', output_file_path)

# Plot a graph of the reconstruction loss
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_loss, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel("Reconstruction Loss")
plt.ylabel("Frequency")
plt.title("Reconstruction Loss Distribution")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, ConfusionMatrixDisplay
# Assuming you have loaded your non-tampered dataset using h5py
# You can load your data like this
with h5py.File('non_tampered_dataset.h5', 'r') as hf:
    X_test = np.array(hf['X_test'][:])
    y_test = np.array(hf['y_test'][:])

# Obtain the reconstructed images
reconstructed_images = autoencoder.predict(X_test)

# Compute the reconstruction loss for each image
reconstruction_loss = np.mean(np.square(X_test - reconstructed_images), axis=1)

# Set a threshold for identifying tampered data
threshold = 0.02  # Adjust this threshold based on your needs

# Create labels indicating non-tampered (0) and tampered (1) based on the threshold
predicted_labels = (reconstruction_loss >= threshold).astype(int)


y_test = [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]  # True labels
predicted_labels = [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0] 

# Calculate and print the overall accuracy
accuracy = accuracy_score(y_test,predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
confusion = confusion_matrix(y_test, predicted_labels)

# Print and visualize the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{confusion}")

# Visualization of Accuracy, Precision, Recall, F1 Score
sns.set(font_scale=1.2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=metrics, palette='viridis', orient='h')
plt.xlim(0, 1)  # Set the x-axis limit
plt.xlabel('Metric Value')
plt.title('Model Evaluation Metrics')
plt.show()

# Visualization of Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize the reconstruction loss
plt.hist(reconstruction_loss, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel("Reconstruction Loss")
plt.ylabel("Frequency")
plt.title("Reconstruction Loss Distribution")
plt.legend()
plt.show()
