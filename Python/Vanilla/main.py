import numpy as np
import time

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        np.random.seed(42)

        self.weights.append(np.random.uniform(
            low=-1,
            high=1,
            size=(hidden_size, input_size)
        ).tolist())
        self.biases.append([0.0] * hidden_size)

        for _ in range(num_hidden_layers - 1):
            self.weights.append(np.random.uniform(
                low=-1,
                high=1,
                size=(hidden_size, hidden_size)
            ).tolist())
            self.biases.append([0.0] * hidden_size)

        self.weights.append(np.random.uniform(
            low=-1,
            high=1,
            size=(output_size, hidden_size)
        ).tolist())
        self.biases.append([0.0] * output_size)


    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

    def predict(self, input_layer):
        # Forward pass
        current = input_layer
        for layer in range(self.num_hidden_layers):
                next_layer = []

                for j in range(self.hidden_size):
                    sum_val = self.biases[layer][j]
                    
                    for k in range(len(current)):
                        sum_val += current[k] * self.weights[layer][j][k]
                    
                    next_layer.append(self.relu(sum_val))
                current = next_layer


        # Output layer
        output = 0.0
        for j in range(self.hidden_size):
            output += current[j] * self.weights[self.num_hidden_layers][0][j]
        output += self.biases[self.num_hidden_layers][0]

        return output

    def train(self, inputs, targets):
        for input_layer, target in zip(inputs, targets):
            # Forward pass

            layer_outputs = [input_layer]
            current = input_layer

            for layer in range(self.num_hidden_layers):
                next_layer = []

                for j in range(self.hidden_size):
                    sum_val = self.biases[layer][j]
                    
                    for k in range(len(current)):
                        sum_val += current[k] * self.weights[layer][j][k]
                    
                    next_layer.append(self.relu(sum_val))
                current = next_layer
                layer_outputs.append(current)

            # Output layer
            output = 0.0
            for j in range(self.hidden_size):
                output += current[j] * self.weights[self.num_hidden_layers][0][j]
            output += self.biases[self.num_hidden_layers][0]
            layer_outputs.append(output)

            output_delta = 2 * (output - target)

            # Update weights and biases
            for j in range(self.hidden_size):
                self.weights[self.num_hidden_layers][0][j] -= self.learning_rate * output_delta * layer_outputs[self.num_hidden_layers - 1][j]
            self.biases[self.num_hidden_layers][0] -= self.learning_rate * output_delta

            delta = [0.0] * self.hidden_size
            for layer in range(self.num_hidden_layers - 1, -1, -1):
                next_delta = [0.0] * self.hidden_size

                for j in range(self.hidden_size):
                    error = sum(delta[k] * self.weights[layer+1][k][j] for k in range(len(delta))) if layer < self.num_hidden_layers - 1 else output_delta * self.weights[layer + 1][0][j]
                    delta[j] = error * self.relu_derivative(layer_outputs[layer + 1][j])

                    for k in range(len(layer_outputs)):
                        self.weights[layer][j][k] -= self.learning_rate * delta[j] * layer_outputs[layer][k]
                    self.biases[layer][j] -= self.learning_rate * delta[j]

                delta = next_delta

def main():
    start_time = time.time()

    inputSize = 5
    numHiddenLayers = 2
    hiddenLayerSize = 4
    outputSize = 1
    learning_rate = 0.02
    epochs = 1000

    x = np.linspace(0, 100)
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
    # go and Cpp brain right there
    # all done, but she's slow