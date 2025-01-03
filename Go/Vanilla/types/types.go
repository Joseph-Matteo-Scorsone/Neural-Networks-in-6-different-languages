package types

import (
	"NeuralNet/utils"
	"math/rand"
)

type NeuralNetwork struct {
	inputSize       int
	hiddenSize      int
	outputSize      int
	numHiddenLayers int
	learningRate    float64
	weights         [][][]float64
	biases          [][]float64
	biasOutput      float64
}

// Constructor function
func NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers int, learningRate float64) *NeuralNetwork {
	r := rand.New(rand.NewSource(42)) // seed for replicability
	nn := &NeuralNetwork{
		inputSize:       inputSize,
		hiddenSize:      hiddenSize,
		outputSize:      outputSize,
		numHiddenLayers: numHiddenLayers,
		learningRate:    learningRate,
		biasOutput:      rand.Float64(),
	}

	// init weights and biases
	nn.weights = make([][][]float64, numHiddenLayers+1)
	nn.biases = make([][]float64, numHiddenLayers+1)

	// input layer
	nn.weights[0] = make([][]float64, hiddenSize)
	nn.biases[0] = make([]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		nn.weights[0][i] = make([]float64, inputSize)
		nn.biases[0][i] = r.Float64()

		for j := 0; j < inputSize; j++ {
			nn.weights[0][i][j] = r.Float64()
		}
	}

	// Hiden layers

	for layer := 1; layer < numHiddenLayers; layer++ {
		nn.weights[layer] = make([][]float64, hiddenSize)
		nn.biases[layer] = make([]float64, hiddenSize)

		for i := 0; i < hiddenSize; i++ {
			nn.weights[layer][i] = make([]float64, hiddenSize)
			nn.biases[layer][i] = rand.Float64()

			for j := 0; j < hiddenSize; j++ {
				nn.weights[layer][i][j] = r.Float64()
			}
		}
	}

	// output layer
	nn.weights[numHiddenLayers] = make([][]float64, outputSize)
	nn.biases[numHiddenLayers] = make([]float64, outputSize)
	for i := 0; i < outputSize; i++ {
		nn.weights[numHiddenLayers][i] = make([]float64, hiddenSize)
		nn.biases[numHiddenLayers][i] = r.Float64()

		for j := 0; j < hiddenSize; j++ {
			nn.weights[numHiddenLayers][i][j] = r.Float64()
		}
	}

	return nn
}

// Train function to do back propagation
func (nn *NeuralNetwork) Train(input []float64, target float64) {
	layerOutputs := [][]float64{input}
	current := input

	//Forward pass
	for layer := 0; layer < nn.numHiddenLayers; layer++ {
		nextLayer := make([]float64, nn.hiddenSize)

		for j := 0; j < nn.hiddenSize; j++ {
			sum := nn.biases[layer][j]

			for k := 0; k < len(current); k++ {
				sum += current[k] * nn.weights[layer][j][k]
			}
			nextLayer[j] = utils.ReLU(sum)
		}
		current = nextLayer
		layerOutputs = append(layerOutputs, current)
	}

	//Output layer
	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += current[j] * nn.weights[nn.numHiddenLayers][0][j] // 0 for 1 output neuron
	}
	layerOutputs = append(layerOutputs, []float64{output})

	// backward pass
	outputDelta := utils.MSEDeriv(target, output)

	// update output layer
	for j := 0; j < nn.hiddenSize; j++ {
		nn.weights[nn.numHiddenLayers][0][j] -= nn.learningRate * outputDelta * layerOutputs[nn.numHiddenLayers][j]
	}

	nn.biasOutput -= nn.learningRate * outputDelta

	// back prop through layers
	delta := make([]float64, nn.hiddenSize)
	for layer := nn.numHiddenLayers - 1; layer >= 0; layer-- {
		nextDelta := make([]float64, nn.hiddenSize)

		for j := 0; j < nn.hiddenSize; j++ {
			err := 0.0

			if layer == nn.numHiddenLayers-1 { // special case for last layer
				err = outputDelta * nn.weights[layer+1][0][j]
			} else { // regular hidden layer
				for k := 0; k < len(delta); k++ {
					err += delta[k] * nn.weights[layer+1][k][j]
				}
			}

			delta[j] = err * utils.ReLUDeriv(layerOutputs[layer+1][j])

			// update weights and biases
			for k := 0; k < len(layerOutputs[layer]); k++ {
				nn.weights[layer][j][k] -= nn.learningRate * delta[j] * layerOutputs[layer][k]
			}
			nn.biases[layer][j] -= nn.learningRate * delta[j]
		}
		delta = nextDelta
	}
}

// Predict is just the forward pass, no back prop
func (nn *NeuralNetwork) Predict(input []float64) float64 {

	current := input

	//Forward pass
	for layer := 0; layer < nn.numHiddenLayers; layer++ {
		nextLayer := make([]float64, nn.hiddenSize)

		for j := 0; j < nn.hiddenSize; j++ {
			sum := nn.biases[layer][j]

			for k := 0; k < len(current); k++ {
				sum += current[k] * nn.weights[layer][j][k]
			}
			nextLayer[j] = utils.ReLU(sum)
		}
		current = nextLayer
	}

	//Output layer
	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += current[j] * nn.weights[nn.numHiddenLayers][0][j] // 0 for 1 output neuron
	}

	return output
}
