import h5py
import requests
import json
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from fl_mnist_implementation_tutorial_utils import *

# Load the MNIST dataset using TensorFlow
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()


with h5py.File(r'/Users/radalaksmi.a.in/Downloads/package-lock/pbft/non_tampered_dataset.h5', 'r') as hf:
    X_train = np.array(hf['X_train'][:])
    y_train = np.array(hf['y_train'][:])
    X_test = np.array(hf['X_test'][:])
    y_test = np.array(hf['y_test'][:])

# Preprocess the data (normalize and reshape)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# /////////////////////
# datasets = []

# ///////////////////////

# Create clients
# clients = create_clients(X_train, y_train, num_clients=5, initial='client')

clients = create_clients(X_train, y_train, num_clients=3, initial='client')

# Process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# Initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(784, 10)

test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
# Commence global training loop
# comms_round = 10
comms_round=100

local_node_accuracies = {node_id: [] for node_id in clients_batched.keys()}
global_accuracies = []
global_losses = []
local_weights_list = []
global_weights_list = []
# local_losses = {}

for comm_round in range(comms_round):
    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # Initial list to collect local model weights after scaling
    scaled_local_weight_list = list()

    # Randomize client data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)

    i=0
    # Loop through each client and create a new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(learning_rate=0.01, momentum=0.9),
                            metrics=['accuracy'])

        # Set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # Fit local model with client's data
        history=local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # Scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        #########################################################

          # Compute accuracy for the local node
        acc, _ = test_model(X_test, y_test, local_model, comm_round)
        local_node_accuracies[client].append(acc)
          # Collect the loss values
        # local_losses[client] = history.history['loss']

        #######################################################

        # Clear session to free memory after each communication round
        K.clear_session()

         # Save the local model to a file
        local_model.save(f'local_model_{client}.h5')

        # # Upload the model to the server
        url = f'http://127.0.0.1:300{i}/transact'
        files = {'file': open(f'local_model_{client}.h5', 'rb')}
        response = requests.post(url, files=files)

        # Print the response from the server
        print('Model upload response:', response.text)
        i=i+1



    # Get the average over all the local models by taking the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # Store the local and global weights in the respective lists
    local_weights_list.append(scaled_local_weight_list)
    global_weights_list.append(average_weights)


    # Update the global model
    global_model.set_weights(average_weights)

    # Test global model and print out metrics after each communications round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        global_accuracies.append(global_acc)
        global_losses.append(global_loss)
    global_model.save('global_model.h5')


# SGD Model
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  metrics=['accuracy'])

# Fit the SGD training data to the model
SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

# Test the SGD global model and print out metrics
for (X_test, Y_test) in test_batched:
    SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)

# for node_id, accuracies in local_node_accuracies.items():
#     plt.plot(range(comms_round), accuracies, label=f"Node {node_id}")

# plt.xlabel("Communication Round")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# for client, losses in local_losses.items():
#     plt.plot(losses, label=f"Local Node {client}")

# plt.xlabel('Communication Round')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Visualize global model accuracy and loss in each round
plt.figure()
plt.plot(range(comms_round), global_accuracies, label="Global Accuracy")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(comms_round), global_losses, label="Global Loss")
plt.xlabel("Communication Round")
plt.ylabel("Loss")
plt.legend()
plt.show()



# from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Assuming you have predictions and true labels for the global model
# global_model_predictions = global_model.predict(X_test)
# true_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels back to integers

# # Calculate accuracy
# accuracy = accuracy_score(true_labels, np.argmax(global_model_predictions, axis=1))
# print(f'Accuracy: {accuracy}')

# # Calculate precision, recall, and F1 score
# precision = precision_score(true_labels, np.argmax(global_model_predictions, axis=1), average='weighted')
# recall = recall_score(true_labels, np.argmax(global_model_predictions, axis=1), average='weighted')
# f1 = f1_score(true_labels, np.argmax(global_model_predictions, axis=1), average='weighted')

# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1}')

# # Calculate and display the confusion matrix
# conf_matrix = confusion_matrix(true_labels, np.argmax(global_model_predictions, axis=1))
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()
