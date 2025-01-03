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
        var rng = Math.random;
        this.weights = [];
        this.biases = [];
        // Initialize weights and biases
        this.weights.push(__spreadArray([], Array(hiddenSize), true).map(function () { return __spreadArray([], Array(inputSize), true).map(function () { return rng(); }); }));
        this.biases.push(__spreadArray([], Array(hiddenSize), true).map(function () { return rng(); }));
        for (var i = 0; i < numHiddenLayers - 1; i++) {
            this.weights.push(__spreadArray([], Array(hiddenSize), true).map(function () { return __spreadArray([], Array(hiddenSize), true).map(function () { return rng(); }); }));
            this.biases.push(__spreadArray([], Array(hiddenSize), true).map(function () { return rng(); }));
        }
        this.weights.push(__spreadArray([], Array(outputSize), true).map(function () { return __spreadArray([], Array(hiddenSize), true).map(function () { return rng(); }); }));
        this.biases.push(__spreadArray([], Array(outputSize), true).map(function () { return rng(); }));
    }
    NeuralNetwork.prototype.relu = function (x) {
        return Math.max(0, x);
    };
    NeuralNetwork.prototype.reluDerivative = function (x) {
        return x > 0 ? 1 : 0;
    };
    NeuralNetwork.prototype.predict = function (inputLayer) {
        var current = inputLayer;
        for (var layer = 0; layer < this.numHiddenLayers; layer++) {
            var nextLayer = [];
            for (var j = 0; j < this.hiddenSize; j++) {
                var sumVal = this.biases[layer][j];
                for (var k = 0; k < current.length; k++) {
                    sumVal += current[k] * this.weights[layer][j][k];
                }
                nextLayer.push(this.relu(sumVal));
            }
            current = nextLayer;
        }
        // output layer
        var output = 0.0;
        for (var j = 0; j < this.hiddenSize; j++) {
            output += current[j] * this.weights[this.numHiddenLayers][0][j];
        }
        output += this.biases[this.numHiddenLayers][0];
        return output;
    };
    NeuralNetwork.prototype.train = function (inputs, targets) {
        var _this = this;
        var _loop_1 = function (i) {
            var inputLayer = inputs[i];
            var target = targets[i];
            // forward pass
            var layerOutputs = [inputLayer];
            var current = inputLayer;
            for (var layer = 0; layer < this_1.numHiddenLayers; layer++) {
                var nextLayer = [];
                for (var j = 0; j < this_1.hiddenSize; j++) {
                    var sumVal = this_1.biases[layer][j];
                    for (var k = 0; k < current.length; k++) {
                        sumVal += current[k] * this_1.weights[layer][j][k];
                    }
                    nextLayer.push(this_1.relu(sumVal));
                }
                current = nextLayer;
                layerOutputs.push(current);
            }
            // output layer
            var output = 0.0;
            for (var j = 0; j < this_1.hiddenSize; j++) {
                output += current[j] * this_1.weights[this_1.numHiddenLayers][0][j];
            }
            output += this_1.biases[this_1.numHiddenLayers][0];
            layerOutputs.push([output]);
            var outputDelta = 2 * (output - target);
            // update weights and biases
            for (var j = 0; j < this_1.hiddenSize; j++) {
                this_1.weights[this_1.numHiddenLayers][0][j] -= this_1.learningRate * outputDelta * layerOutputs[this_1.numHiddenLayers - 1][j];
            }
            this_1.biases[this_1.numHiddenLayers][0] -= this_1.learningRate * outputDelta;
            var delta = Array(this_1.hiddenSize).fill(0);
            var _loop_2 = function (layer) {
                var nextDelta = Array(this_1.hiddenSize).fill(0);
                var _loop_3 = function (j) {
                    var error = layer < this_1.numHiddenLayers - 1 ? delta.reduce(function (sum, _, k) { return sum + delta[k] * _this.weights[layer + 1][k][j]; }, 0)
                        : outputDelta * this_1.weights[layer + 1][0][j];
                    delta[j] = error * this_1.reluDerivative(layerOutputs[layer + 1][j]);
                    for (var k = 0; k < layerOutputs[layer].length; k++) {
                        this_1.weights[layer][j][k] -= this_1.learningRate * delta[j] * layerOutputs[layer][k];
                    }
                    this_1.biases[layer][j] -= this_1.learningRate * delta[j];
                };
                for (var j = 0; j < this_1.hiddenSize; j++) {
                    _loop_3(j);
                }
                delta = nextDelta;
            };
            for (var layer = this_1.numHiddenLayers - 1; layer >= 0; layer--) {
                _loop_2(layer);
            }
        };
        var this_1 = this;
        for (var i = 0; i < inputs.length; i++) {
            _loop_1(i);
        }
    };
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
    // Sin
    var x = Array.from({ length: 100 }, function (_, i) { return i; });
    var y = x.map(function (val) { return Math.sin(val); });
    // training data
    var inputs = y.slice(0, y.length - inputSize).map(function (_, i) { return y.slice(i, i + inputSize); });
    var targets = y.slice(inputSize);
    //Initialize network
    var nn = new NeuralNetwork(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate);
    // train
    for (var epoch = 0; epoch < epochs; epoch++) {
        nn.train(inputs, targets);
    }
    // Predict and output
    var lastInput = inputs[inputs.length - 1];
    var prediction = nn.predict(lastInput);
    var actual = y[y.length - 1];
    var err = Math.abs(prediction - actual);
    console.log("Prediction for next value: ".concat(prediction.toFixed(2)));
    console.log("Actual next value: ".concat(actual.toFixed(2)));
    console.log("Error: ".concat(err.toFixed(2)));
    var end = Date.now();
    var executionTime = end - start;
    console.log("Elapsed time: ".concat((executionTime / 1000), " Seconds"));
}
main();
