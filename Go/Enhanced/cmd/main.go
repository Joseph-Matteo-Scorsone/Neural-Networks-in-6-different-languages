package main

import (
	"NeuralNet/types"
	"fmt"
	"math"
	"sync"
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

	batchSize := 10
	numBtaches := len(inputs) / batchSize

	// Training loop with batches
	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup
		for batch := 0; batch < numBtaches; batch ++ {
			wg.Add(1)
			go func (batch int) {
				defer wg.Done()
				startIdx := batch * batchSize
				endIdx := startIdx + batchSize
				for i := startIdx; i < endIdx; i++ {
					net.Train(inputs[i], targets[i])
				}
			}(batch)
		}
		wg.Wait()
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
	fmt.Printf("Elapsed time: %.6f seconds\n", elapsed.Seconds())
}
// fixes. Old code. Pretty dang fast. Modify for batch training. Just main.go
// thats all
// still very fast. a bit slower. Maybe spawning go routines can slow it down
// :)