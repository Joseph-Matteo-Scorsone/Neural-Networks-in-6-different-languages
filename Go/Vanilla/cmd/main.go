package main

import (
	"NeuralNet/types"
	"fmt"
	"math"
	"time"
)

func main() {
	// Start timing
	start := time.Now()

	inputSize := 5
	numHiddenLayers := 2
	hiddenSize := 4
	outputSize := 1
	learningRate := 0.02
	epochs := 1000

	net := types.NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers, learningRate)

	// Sin data for predicting
	sinData := []float64{}
	dataLength := 100
	for i := 0; i < dataLength; i++ {
		value := math.Sin(float64(i))
		sinData = append(sinData, value)
	}

	// training data
	inputs := [][]float64{}
	targets := []float64{}
	for i := inputSize; i < dataLength; i++ {
		input := []float64{sinData[i-inputSize], sinData[i-1]}
		inputs = append(inputs, input)
		targets = append(targets, sinData[i])
	}

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(inputs); i++ {
			net.Train(inputs[i], targets[i])
		}
	}

	//Actual value
	actual := math.Sin(float64(dataLength))
	fmt.Printf("Actual next: %.2f\n", actual)

	// test input
	testInput := []float64{sinData[dataLength-inputSize], sinData[dataLength-1]}
	prediction := net.Predict(testInput)
	fmt.Printf("Predicted: %.2f\n", prediction)

	err := math.Abs(prediction - actual)
	fmt.Printf("Error: %.2f\n", err)

	// Done RRAAAHHHHHHHHH

	// End timing and output
	elapsed := time.Since(start)
	fmt.Printf("Execution time: %.6f seconds\n", elapsed.Seconds())
}
