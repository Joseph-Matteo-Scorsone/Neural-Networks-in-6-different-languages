import numpy as np
import time

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        
        np.random.seed(42)

        # Input to first hidden layer
        self.weights = [
            np.random.uniform(-1, 1, (hidden_size, input_size))
        ]
        self.biases = [np.zeros(hidden_size)]

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(
                np.random.uniform(-1, 1, (hidden_size, hidden_size))
            )
            self.biases.append(np.zeros(hidden_size))


        # Output layer
        self.weights.append(
            np.random.uniform(-1, 1, (output_size, hidden_size))
        )
        self.biases.append(np.zeros(output_size))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


    def predict(self, input_layer: np.ndarray) -> float:
        current = input_layer

        # Forward pass through hidden layers
        for layer in range(self.num_hidden_layers):
            # use np arrays
            current = self.relu(
                np.dot(self.weights[layer], current) + self.biases[layer]
            )

        # Output layer
        output = np.dot(self.weights[-1], current) + self.biases[-1]
        return float(output[0])

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        for input_layer, target in zip(inputs, targets):
            # Forward pass
            layer_outputs = [input_layer]
            current = input_layer

            # Hidden layers
            for layer in range(self.num_hidden_layers):
                current = self.relu(
                    np.dot(self.weights[layer], current) + self.biases[layer]
                )
                layer_outputs.append(current)

            # Output layer
            output = np.dot(self.weights[-1], current) + self.biases[-1]
            layer_outputs.append(output)

            # Backpropagation
            output_delta = 2 * (output - target)

            # Output layer weights and bias update
            self.weights[-1] -= (
                self.learning_rate * output_delta * layer_outputs[-2].reshape(-1, 1).T
            )
            self.biases[-1] -= self.learning_rate * output_delta

            # Hidden layers
            delta = output_delta
            for layer in range(self.num_hidden_layers - 1, -1, -1):
                delta = np.dot(self.weights[layer + 1].T, delta) * \
                        self.relu_derivative(layer_outputs[layer + 1])
               
                # Update weights and biases
                self.weights[layer] -= (
                    self.learning_rate *
                    np.outer(delta, layer_outputs[layer])
                )
                self.biases[layer] -= self.learning_rate * delta

def main():
    start_time = time.time()

    inputSize = 5
    numHiddenLayers = 2
    hiddenLayerSize = 4
    outputSize = 1
    learning_rate = 0.02
    epochs = 1000

    x = np.linspace(0, 100, num=100)
    y = np.sin(x)

    inputs = np.array([y[i:i+inputSize] for i in range(len(y)-inputSize)])
    targets = y[inputSize:]

    nn = NeuralNetwork(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learning_rate)

    for _ in range(epochs):
        nn.train(inputs, targets)

    last_input = inputs[-1]
    prediction = nn.predict(last_input)
    actual = y[-1]

    print(f"Prediction: {prediction:.2f}")
    print(f"Actual: {actual:.2f}")
    print(f"Error: {(prediction - actual):.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
    # I think over a second off