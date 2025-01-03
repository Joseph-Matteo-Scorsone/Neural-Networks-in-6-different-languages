class NeuralNetwork {
    inputSize: number;
    hiddenSize: number;
    outputSize: number;
    numHiddenLayers: number;
    learningRate: number;
    weights: number[][][];
    biases: number[][];

    constructor(inputSize: number, hiddenSize: number, outputSize: number, numHiddenLayers: number, learningRate: number) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.numHiddenLayers = numHiddenLayers;
        this.learningRate = learningRate;

        const rng = Math.random;

        this.weights = [];
        this.biases = [];

        // Initialize weights and biases
        this.weights.push([...Array(hiddenSize)].map(() => [...Array(inputSize)].map(() => rng())));
        this.biases.push([...Array(hiddenSize)].map(() => rng()));
        for (let i = 0; i < numHiddenLayers - 1; i++) {
            this.weights.push([...Array(hiddenSize)].map(() => [...Array(hiddenSize)].map(() => rng())));
            this.biases.push([...Array(hiddenSize)].map(() => rng()));
        } 

        this.weights.push([...Array(outputSize)].map(() => [...Array(hiddenSize)].map(() => rng())));
        this.biases.push([...Array(outputSize)].map(() => rng()));
    }
    
    relu(x: number): number {
        return Math.max(0, x);
    }

    reluDerivative(x: number): number {
        return x > 0 ? 1 : 0;
    }

    predict(inputLayer: number[]): number {
        let current = inputLayer;
        for (let layer = 0; layer < this.numHiddenLayers; layer++) {
            const nextLayer: number[] = [];

            for (let j = 0; j < this.hiddenSize; j++) {
                let sumVal = this.biases[layer][j];

                for (let k = 0; k < current.length; k++) {
                    sumVal += current[k] * this.weights[layer][j][k];
                }

                nextLayer.push(this.relu(sumVal));
            }
            current = nextLayer;
        }

        // output layer
        let output = 0.0;
        for (let j = 0; j < this.hiddenSize; j++) {
            output += current[j] * this.weights[this.numHiddenLayers][0][j];
        }
        output += this.biases[this.numHiddenLayers][0];
        
        return output;
    }

    train(inputs: number[][], targets: number[]) {
        for (let i = 0; i < inputs.length; i++) {
            const inputLayer = inputs[i];
            const target = targets[i];

            // forward pass
            const layerOutputs: number[][] = [inputLayer];
            let current = inputLayer;
            for (let layer = 0; layer < this.numHiddenLayers; layer++) {
                const nextLayer: number[] = [];

                for (let j = 0; j < this.hiddenSize; j++) {
                    let sumVal = this.biases[layer][j];
 
                    for (let k = 0; k < current.length; k++) {
                        sumVal += current[k] * this.weights[layer][j][k];
                    }

                    nextLayer.push(this.relu(sumVal));
                }
                current = nextLayer;
                layerOutputs.push(current);
            }

            // output layer
            let output = 0.0;
            for (let j = 0; j < this.hiddenSize; j++) {
                output += current[j] * this.weights[this.numHiddenLayers][0][j];
            }
            output += this.biases[this.numHiddenLayers][0];
            layerOutputs.push([output]);

            let outputDelta = 2 * (output - target);
            
            // update weights and biases
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weights[this.numHiddenLayers][0][j] -= this.learningRate * outputDelta * layerOutputs[this.numHiddenLayers - 1][j];
            }
            this.biases[this.numHiddenLayers][0] -= this.learningRate * outputDelta;

            let delta = Array(this.hiddenSize).fill(0);
            for (let layer = this.numHiddenLayers - 1; layer >= 0; layer--) {
                const nextDelta: number[] = Array(this.hiddenSize).fill(0);

                for (let j = 0; j < this.hiddenSize; j++) {
                    const error = layer < this.numHiddenLayers - 1 ? delta.reduce((sum, _, k) => sum + delta[k] * this.weights[layer + 1][k][j], 0)
                    : outputDelta * this.weights[layer + 1][0][j];
                    delta[j] = error * this.reluDerivative(layerOutputs[layer + 1][j]);

                    for (let k = 0; k < layerOutputs[layer].length; k++) {
                        this.weights[layer][j][k] -= this.learningRate * delta[j] * layerOutputs[layer][k];
                    }
                    this.biases[layer][j] -= this.learningRate * delta[j];
                }
                delta = nextDelta;
            }
        }
    }
}

function main() {
    const start = Date.now();

    const inputSize = 5;
    const numHiddenLayers = 2;
    const hiddenLayerSize = 4;
    const outputSize = 1;
    const learningRate = 0.02;
    const epochs = 1000;

    // Sin
    const x = Array.from({ length: 100}, (_, i) => i);
    const y = x.map(val => Math.sin(val));

    // training data
    const inputs = y.slice(0, y.length - inputSize).map((_, i) => y.slice(i, i + inputSize));
    const targets = y.slice(inputSize);

    //Initialize network
    const nn = new NeuralNetwork(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate);

    // train
    for (let epoch = 0; epoch < epochs; epoch++) {
        nn.train(inputs, targets);
    }

    // Predict and output
    const lastInput = inputs[inputs.length - 1];
    const prediction = nn.predict(lastInput);
    const actual = y[y.length - 1];
    const err = Math.abs(prediction - actual);
    console.log(`Prediction for next value: ${prediction.toFixed(2)}`);
    console.log(`Actual next value: ${actual.toFixed(2)}`);
    console.log(`Error: ${err.toFixed(2)}`);

    const end = Date.now();
    const executionTime = end - start;
    console.log(`Elapsed time: ${(executionTime / 1000)} Seconds`);
}

main()