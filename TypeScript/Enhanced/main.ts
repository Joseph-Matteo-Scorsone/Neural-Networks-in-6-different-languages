class NeuralNetwork {
    private readonly inputSize: number;
    private readonly hiddenSize: number;
    private readonly outputSize: number;
    private readonly numHiddenLayers: number;
    private readonly learningRate: number;
    private readonly weights: Float64Array[];
    private readonly biases: Float64Array[];
    private readonly layerSizes: number[];

    constructor(
        inputSize: number,
        hiddenSize: number,
        outputSize: number,
        numHiddenLayers: number,
        learningRate: number
    ) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.numHiddenLayers = numHiddenLayers;
        this.learningRate = learningRate;

        this.layerSizes = [
            inputSize,
            ...Array(numHiddenLayers).fill(hiddenSize),
            outputSize
        ];

        this.weights = [];
        this.biases = [];

        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            const currentSize = this.layerSizes[i];
            const nextSize = this.layerSizes[i + 1];

            const limit = Math.sqrt(6 / (currentSize + nextSize));
            const weightsLayer = new Float64Array(currentSize * nextSize);
            for (let j = 0; j < weightsLayer.length; j++) {
                weightsLayer[j] = Math.random() * 2 * limit - limit;
            }
            this.weights.push(weightsLayer);
           
            this.biases.push(new Float64Array(nextSize));
        }
    }

    private static readonly relu = (x: number): number => x > 0 ? x : 0;
    private static readonly reluDerivative = (x: number): number => x > 0 ? 1 : 0;

    predict(inputLayer: number[]): number {
        const layerOutputs: Float64Array[] = [new Float64Array(inputLayer)];

        for (let layer = 0; layer < this.numHiddenLayers + 1; layer++) {
            const currentSize = this.layerSizes[layer];
            const nextSize = this.layerSizes[layer + 1];
            const nextLayer = new Float64Array(nextSize);

            for (let j = 0; j < nextSize; j++) {
                let sum = this.biases[layer][j];
                const weightOffset = j * currentSize;
               
                // Unroll the loop for better performance
                const limit = currentSize - (currentSize % 4);
                let k = 0;

                // Process 4 elements at a time
                for (; k < limit; k += 4) {
                    sum += layerOutputs[layer][k] * this.weights[layer][weightOffset + k] +
                          layerOutputs[layer][k + 1] * this.weights[layer][weightOffset + k + 1] +
                          layerOutputs[layer][k + 2] * this.weights[layer][weightOffset + k + 2] +
                          layerOutputs[layer][k + 3] * this.weights[layer][weightOffset + k + 3];
                }

                // Handle remaining elements
                for (; k < currentSize; k++) {
                    sum += layerOutputs[layer][k] * this.weights[layer][weightOffset + k];
                }
               
                nextLayer[j] = layer < this.numHiddenLayers ? NeuralNetwork.relu(sum) : sum;
            }
           
            layerOutputs.push(nextLayer);
        }

        return layerOutputs[layerOutputs.length - 1][0];
    }

    private backwardPass(
        layerOutputs: Float64Array[],
        target: number,
        weightGradients: Float64Array[],
        biasGradients: Float64Array[]
    ): void {
        const output = layerOutputs[layerOutputs.length - 1][0];
        const outputError = 2 * (output - target);
       
        let deltas = new Float64Array(this.hiddenSize);

        // Output layer gradients
        for (let j = 0; j < this.hiddenSize; j++) {
            const lastHiddenOutput = layerOutputs[this.numHiddenLayers][j];
            const gradient = outputError * lastHiddenOutput;
            weightGradients[this.numHiddenLayers][j] += gradient;
        }
        biasGradients[this.numHiddenLayers][0] += outputError;

        // Hidden layers gradients
        for (let layer = this.numHiddenLayers - 1; layer >= 0; layer--) {
            const nextDeltas = new Float64Array(this.hiddenSize);
           
            for (let j = 0; j < this.hiddenSize; j++) {
                const error = layer < this.numHiddenLayers - 1
                    ? deltas.reduce((sum, d, k) => sum + d * this.weights[layer + 1][k * this.hiddenSize + j], 0)
                    : outputError * this.weights[layer + 1][j];
                   
                nextDeltas[j] = error * NeuralNetwork.reluDerivative(layerOutputs[layer + 1][j]);
               
                const weightOffset = j * this.layerSizes[layer];
                for (let k = 0; k < this.layerSizes[layer]; k++) {
                    weightGradients[layer][weightOffset + k] += nextDeltas[j] * layerOutputs[layer][k];
                }
                biasGradients[layer][j] += nextDeltas[j];
            }
           
            deltas = nextDeltas;
        }
    }

    private forwardPass(input: number[]): Float64Array[] {
        const layerOutputs: Float64Array[] = [new Float64Array(input)];
       
        for (let layer = 0; layer < this.numHiddenLayers + 1; layer++) {
            const currentSize = this.layerSizes[layer];
            const nextSize = this.layerSizes[layer + 1];
            const nextLayer = new Float64Array(nextSize);

            for (let j = 0; j < nextSize; j++) {
                let sum = this.biases[layer][j];
                const weightOffset = j * currentSize;
               
                for (let k = 0; k < currentSize; k++) {
                    sum += layerOutputs[layer][k] * this.weights[layer][weightOffset + k];
                }
                nextLayer[j] = layer < this.numHiddenLayers ? NeuralNetwork.relu(sum) : sum;
            }
              
            layerOutputs.push(nextLayer);
        }
        return layerOutputs;
    }


    private trainMiniBatch(
        batchInputs: number[][],
        batchTargets: number[],
        weightGradients: Float64Array[],
        biasGradients: Float64Array[]
    ): void {
        // Reset gradients
        weightGradients.forEach(g => g.fill(0));
        biasGradients.forEach(g => g.fill(0));
        
        for (let i = 0; i < batchInputs.length; i++) {
            // Forward pass (reuse predict logic)
            const layerOutputs = this.forwardPass(batchInputs[i]);
           
            // Backward pass
            this.backwardPass(
                layerOutputs,
                batchTargets[i],
                weightGradients,
                biasGradients
            );
        }

        // Apply gradients with momentum
        const batchScale = this.learningRate / batchInputs.length;
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] -= weightGradients[i][j] * batchScale;
            }
            for (let j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] -= biasGradients[i][j] * batchScale;
            }
        }

    }

    train(inputs: number[][], targets: number[]) {
        // Pre-allocate arrays for gradients
        const weightGradients: Float64Array[] = this.weights.map(w => new Float64Array(w.length));
        const biasGradients: Float64Array[] = this.biases.map(b => new Float64Array(b.length));

        // Mini-batch processing
        const batchSize = Math.min(32, inputs.length);
       
        for (let i = 0; i < inputs.length; i += batchSize) {
            const batchEnd = Math.min(i + batchSize, inputs.length);
            this.trainMiniBatch(
                inputs.slice(i, batchEnd),
                targets.slice(i, batchEnd),
                weightGradients,
                biasGradients
            );
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

    // Generate sin
    const x = new Float64Array(100);
    const y = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        x[i] = i;
        y[i] = Math.sin(i);
    }

    // Prepare training data using a sliding window
    const inputs: number[][] = [];
    const targets: number[] = [];
   
    for (let i = 0; i < y.length - inputSize; i++) {
        inputs.push(Array.from(y.subarray(i, i + inputSize)));
        targets.push(y[i + inputSize]);
    }

    // Initialize the network
    const nn = new NeuralNetwork(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate);

    // Train the network
    for (let epoch = 0; epoch < epochs; epoch++) {
        nn.train(inputs, targets);
    }

    // Predict and print the next value
    const lastInput = inputs[inputs.length - 1];
    const prediction = nn.predict(lastInput);
    const actual = y[y.length - 1];
    const err = prediction - actual;
    console.log(`Prediction for next value: ${prediction.toFixed(2)}`);
    console.log(`Actual next value: ${actual.toFixed(2)}`);
    console.log(`Error: ${err.toFixed(2)}`);

    const end = Date.now();
    const executionTime = end - start;
    console.log(`Elapsed time: ${(executionTime / 1000)} Seconds`);
}

main()