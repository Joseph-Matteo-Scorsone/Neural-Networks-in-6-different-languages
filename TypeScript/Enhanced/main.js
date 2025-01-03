var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var NeuralNetwork = /** @class */ (function () {
    function NeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers, learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.numHiddenLayers = numHiddenLayers;
        this.learningRate = learningRate;
        this.layerSizes = __spreadArray(__spreadArray([
            inputSize
        ], Array(numHiddenLayers).fill(hiddenSize), true), [
            outputSize
        ], false);
        this.weights = [];
        this.biases = [];
        for (var i = 0; i < this.layerSizes.length - 1; i++) {
            var currentSize = this.layerSizes[i];
            var nextSize = this.layerSizes[i + 1];
            var limit = Math.sqrt(6 / (currentSize + nextSize));
            var weightsLayer = new Float64Array(currentSize * nextSize);
            for (var j = 0; j < weightsLayer.length; j++) {
                weightsLayer[j] = Math.random() * 2 * limit - limit;
            }
            this.weights.push(weightsLayer);
            this.biases.push(new Float64Array(nextSize));
        }
    }
    NeuralNetwork.prototype.predict = function (inputLayer) {
        var layerOutputs = [new Float64Array(inputLayer)];
        for (var layer = 0; layer < this.numHiddenLayers + 1; layer++) {
            var currentSize = this.layerSizes[layer];
            var nextSize = this.layerSizes[layer + 1];
            var nextLayer = new Float64Array(nextSize);
            for (var j = 0; j < nextSize; j++) {
                var sum = this.biases[layer][j];
                var weightOffset = j * currentSize;
                // Unroll the loop for better performance
                var limit = currentSize - (currentSize % 4);
                var k = 0;
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
    };
    NeuralNetwork.prototype.backwardPass = function (layerOutputs, target, weightGradients, biasGradients) {
        var _this = this;
        var output = layerOutputs[layerOutputs.length - 1][0];
        var outputError = 2 * (output - target);
        var deltas = new Float64Array(this.hiddenSize);
        // Output layer gradients
        for (var j = 0; j < this.hiddenSize; j++) {
            var lastHiddenOutput = layerOutputs[this.numHiddenLayers][j];
            var gradient = outputError * lastHiddenOutput;
            weightGradients[this.numHiddenLayers][j] += gradient;
        }
        biasGradients[this.numHiddenLayers][0] += outputError;
        var _loop_1 = function (layer) {
            var nextDeltas = new Float64Array(this_1.hiddenSize);
            var _loop_2 = function (j) {
                var error = layer < this_1.numHiddenLayers - 1
                    ? deltas.reduce(function (sum, d, k) { return sum + d * _this.weights[layer + 1][k * _this.hiddenSize + j]; }, 0)
                    : outputError * this_1.weights[layer + 1][j];
                nextDeltas[j] = error * NeuralNetwork.reluDerivative(layerOutputs[layer + 1][j]);
                var weightOffset = j * this_1.layerSizes[layer];
                for (var k = 0; k < this_1.layerSizes[layer]; k++) {
                    weightGradients[layer][weightOffset + k] += nextDeltas[j] * layerOutputs[layer][k];
                }
                biasGradients[layer][j] += nextDeltas[j];
            };
            for (var j = 0; j < this_1.hiddenSize; j++) {
                _loop_2(j);
            }
            deltas = nextDeltas;
        };
        var this_1 = this;
        // Hidden layers gradients
        for (var layer = this.numHiddenLayers - 1; layer >= 0; layer--) {
            _loop_1(layer);
        }
    };
    NeuralNetwork.prototype.forwardPass = function (input) {
        var layerOutputs = [new Float64Array(input)];
        for (var layer = 0; layer < this.numHiddenLayers + 1; layer++) {
            var currentSize = this.layerSizes[layer];
            var nextSize = this.layerSizes[layer + 1];
            var nextLayer = new Float64Array(nextSize);
            for (var j = 0; j < nextSize; j++) {
                var sum = this.biases[layer][j];
                var weightOffset = j * currentSize;
                for (var k = 0; k < currentSize; k++) {
                    sum += layerOutputs[layer][k] * this.weights[layer][weightOffset + k];
                }
                nextLayer[j] = layer < this.numHiddenLayers ? NeuralNetwork.relu(sum) : sum;
            }
            layerOutputs.push(nextLayer);
        }
        return layerOutputs;
    };
    NeuralNetwork.prototype.trainMiniBatch = function (batchInputs, batchTargets, weightGradients, biasGradients) {
        // Reset gradients
        weightGradients.forEach(function (g) { return g.fill(0); });
        biasGradients.forEach(function (g) { return g.fill(0); });
        for (var i = 0; i < batchInputs.length; i++) {
            // Forward pass (reuse predict logic)
            var layerOutputs = this.forwardPass(batchInputs[i]);
            // Backward pass
            this.backwardPass(layerOutputs, batchTargets[i], weightGradients, biasGradients);
        }
        // Apply gradients with momentum
        var batchScale = this.learningRate / batchInputs.length;
        for (var i = 0; i < this.weights.length; i++) {
            for (var j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] -= weightGradients[i][j] * batchScale;
            }
            for (var j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] -= biasGradients[i][j] * batchScale;
            }
        }
    };
    NeuralNetwork.prototype.train = function (inputs, targets) {
        // Pre-allocate arrays for gradients
        var weightGradients = this.weights.map(function (w) { return new Float64Array(w.length); });
        var biasGradients = this.biases.map(function (b) { return new Float64Array(b.length); });
        // Mini-batch processing
        var batchSize = Math.min(32, inputs.length);
        for (var i = 0; i < inputs.length; i += batchSize) {
            var batchEnd = Math.min(i + batchSize, inputs.length);
            this.trainMiniBatch(inputs.slice(i, batchEnd), targets.slice(i, batchEnd), weightGradients, biasGradients);
        }
    };
    NeuralNetwork.relu = function (x) { return x > 0 ? x : 0; };
    NeuralNetwork.reluDerivative = function (x) { return x > 0 ? 1 : 0; };
    return NeuralNetwork;
}());
function main() {
    var start = Date.now();
    var inputSize = 5;
    var numHiddenLayers = 2;
    var hiddenLayerSize = 4;
    var outputSize = 1;
    var learningRate = 0.02;
    var epochs = 1000;
    // Generate sin
    var x = new Float64Array(100);
    var y = new Float64Array(100);
    for (var i = 0; i < 100; i++) {
        x[i] = i;
        y[i] = Math.sin(i);
    }
    // Prepare training data using a sliding window
    var inputs = [];
    var targets = [];
    for (var i = 0; i < y.length - inputSize; i++) {
        inputs.push(Array.from(y.subarray(i, i + inputSize)));
        targets.push(y[i + inputSize]);
    }
    // Initialize the network
    var nn = new NeuralNetwork(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate);
    // Train the network
    for (var epoch = 0; epoch < epochs; epoch++) {
        nn.train(inputs, targets);
    }
    // Predict and print the next value
    var lastInput = inputs[inputs.length - 1];
    var prediction = nn.predict(lastInput);
    var actual = y[y.length - 1];
    var err = prediction - actual;
    console.log("Prediction for next value: ".concat(prediction.toFixed(2)));
    console.log("Actual next value: ".concat(actual.toFixed(2)));
    console.log("Error: ".concat(err.toFixed(2)));
    var end = Date.now();
    var executionTime = end - start;
    console.log("Elapsed time: ".concat((executionTime / 1000), " Seconds"));
}
main();
