#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

class NeuralNetwork{
    public:
        // constructor and public API
        NeuralNetwork(const int inputSize, const int hiddenLayerSize, const int outputSize, const int numHiddenLayers, const float learningRate)
            : inputSize(inputSize), hiddenLayerSize(hiddenLayerSize), outputSize(outputSize), numHiddenLayers(numHiddenLayers) {
                InitializeRandomWeightsAndBiases();
            }

        void Train(const std::vector<float> input, const float target) {
            // could put an assert here but it's ok
            _train(input, target);
        }

        float Predict(const std::vector<float> input) {
            return _predict(input);
        } 

    private:
        // Variables
        int inputSize, hiddenLayerSize, outputSize, numHiddenLayers;
        float biasOutput, learningRate;

        std::vector<std::vector<std::vector<float>>> weights;
        std::vector<std::vector<float>> biases;

        // train funciton to do back propagation.
        void _train(const std::vector<float> input, const float target) {
            std::vector<float> current = input;
            std::vector<std::vector<float>> layer_outputs;
            layer_outputs.push_back(current);

            //Forward pass. This could probably get its own function, but its ok
            for (int i = 0; i < numHiddenLayers; i++) {
                std::vector<float> nextLayer(hiddenLayerSize, 0.0f);
                for (int j = 0; j < hiddenLayerSize; j++) {
                    float sum = biases[i][j];
                    
                    for (int k = 0; k < current.size(); k++) {
                        sum += current[k] * weights[i][j][k];
                    }
                    nextLayer[j] = ReLU(sum);
                }
                current = nextLayer;
                layer_outputs.push_back(current);
            }

            // foward pass through just the output layer
            float output = biasOutput;
            for (int i = 0; i < hiddenLayerSize; i++) {
                output += current[i] * weights[numHiddenLayers][0][i]; // hard code 0 because we assume output size will always be 1
            }

            // check error
            float error = MSE(target, output);
            float d_error = MSEDeriv(target, output);

            // Back propagate the output 
            float output_delta = d_error * ReLUDeriv(output);

            // update weights and biases of output neuron
            for (int i = 0; i < hiddenLayerSize; i++) {
                weights[numHiddenLayers][0][i] -= learningRate * output_delta * layer_outputs[numHiddenLayers][i];
            }

            biasOutput -= learningRate * output_delta;

            // back prop through the layers now
            std::vector<float> delta(hiddenLayerSize, 0.0f);
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                std::vector<float> nextDelta(hiddenLayerSize, 0.0f);
                
                for (int j = 0; j < hiddenLayerSize; j++) {
                    float error = 0.0f;
                    for (int k = 0; k < weights[layer + 1].size(); k++) {
                        error += delta[k] * weights[layer + 1][k][j]; // k then j here
                    }

                    float delta_j = error * ReLUDeriv(layer_outputs[layer + 1][j]);
                    nextDelta[j] = delta_j;

                    // update weights and biases for current neuron
                    for (int k = 0; k < layer_outputs[layer].size(); k++) {
                        weights[layer][j][k] -= learningRate * delta_j * layer_outputs[layer][k];
                    }
                    
                    delta = nextDelta;
                }
            }
        }

        // Prediction function is like a forward pass without the back prop
        float _predict(const std::vector<float> input) {
            
            std::vector<float> current = input;

            ///Forward pass. This could probably get its own function, but its ok
            for (int i = 0; i < numHiddenLayers; i++) {
                std::vector<float> nextLayer(hiddenLayerSize, 0.0f);
                for (int j = 0; j < hiddenLayerSize; j++) {
                    float sum = biases[i][j];
                    
                    for (int k = 0; k < current.size(); k++) {
                        sum += current[k] * weights[i][j][k];
                    }
                    nextLayer[j] = ReLU(sum);
                }
                current = nextLayer;
            }

            // foward pass through just the output layer
            float output = biasOutput;
            for (int i = 0; i < hiddenLayerSize; i++) {
                output += current[i] * weights[numHiddenLayers][0][i]; // hard code 0 because we assume output size will always be 1
            }
            return output;
        }
        
        // activation functions
        float ReLU(float x) {
            if (x > 0) {
                return x;
            }
            return 0;
        }

        float ReLUDeriv(float x) {
            if (x > 0) {
                return 1;
            }
            return 0;
        }

        float MSE(float y, float yPred) {
            return 0.5 * (y - yPred) * (y - yPred);
        }

        float MSEDeriv(float y, float yPred) {
            return yPred - y;
        }
        
        // initialize function to be called in constructor
        void InitializeRandomWeightsAndBiases() {
            std::random_device rd;
            std::mt19937 gen(42);
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            //input layer up to first layer
            weights.push_back(std::vector<std::vector<float>>());
            biases.push_back(std::vector<float>());

            for (int i = 0; i < hiddenLayerSize; i++) {
                weights[0].push_back(std::vector<float>());
                biases[0].push_back(dis(gen));
                for (int j = 0; j < inputSize; j++) {
                    weights[0][i].push_back(dis(gen));
                }
            }

            // Hidden layers
            for (int layer = 1; layer < numHiddenLayers; layer++) {
                weights.push_back(std::vector<std::vector<float>>());
                biases.push_back(std::vector<float>());

                for (int i = 0; i < hiddenLayerSize; i++) {
                    weights[layer].push_back(std::vector<float>());
                    biases[layer].push_back(dis(gen));

                    for (int j = 0; j < hiddenLayerSize; j++) {
                        weights[layer][i].push_back(dis(gen));
                    }
                }
            }

            // output layer
            weights.push_back(std::vector<std::vector<float>>());
            biases.push_back(std::vector<float>());

            for (int i = 0; i < outputSize; i++) {
                weights[numHiddenLayers].push_back(std::vector<float>());
                biases[numHiddenLayers].push_back(dis(gen));

                for (int j = 0; j < hiddenLayerSize; j++) {
                    weights[numHiddenLayers][i].push_back(dis(gen));
                }
            }

            biasOutput = dis(gen);
        }
};

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    const int inputSize = 5;
    const int hiddenLayerSize = 4;
    const int outputSize = 1;
    const int numHiddenLayers = 2;
    const float learningRate = 0.02f;

    NeuralNetwork nn(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate);

    std::vector<float> ts;
    int dataLength = 100;
    std::random_device rd;
    std::mt19937 gen(42);

    // prepare data
    for (int i = 0; i < dataLength; i++) {
        float value = std::sin(i);
        ts.push_back(value);
    }

    // sliding window for training data
    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    for (int i = inputSize; i < dataLength; i++) {
        std::vector<float> input(ts.begin() + i - inputSize, ts.begin() + i);
        inputs.push_back(input);
        targets.push_back(ts[i]);
    }

    // training
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputSize; i++) {
            nn.Train(inputs[i], targets[i]);
        }
    }

    // test it
    const std::vector<float> testInput(ts.end() - inputSize, ts.end());
    float prediciton = nn.Predict(testInput);
    std::cout << "Prediction: " << prediciton << std::endl;
    
    float actual = std::sin(dataLength);
    std::cout << "Actual: " << actual << std::endl;

    float err = std::abs(prediciton - actual);
    std::cout << "Error: " << err << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    return 0;
}
