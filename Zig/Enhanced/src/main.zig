const std = @import("std");

pub fn NeuralNetwork(comptime input_size: comptime_int, comptime hidden_size: comptime_int, comptime hidden_layers: comptime_int) type {
    
    // const vector size for simplicity
    const Vec = @Vector(4, f64);
    const vec_size = 4;
    const hidden_vec_count = hidden_size / vec_size + @as(usize, @intFromBool(hidden_size % vec_size != 0));

    return struct {
        const Self = @This();

        first_layer_weights: [hidden_size][input_size]f64,
        hidden_weights: [hidden_layers - 1][hidden_size][hidden_size]f64,
        output_weights: [hidden_size]f64,
        biases: [hidden_layers][hidden_size]f64,
        bias_output: f64,
        learning_rate: f64,

        pub fn init(learning_rate: f64) Self {
            const seed = 42;
            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();

            var self = Self{
                .first_layer_weights = undefined,
                .hidden_weights = undefined,
                .output_weights = undefined,
                .biases = undefined,
                .bias_output = 0,
                .learning_rate = learning_rate,
            };

            //scalars 
            const first_layer_scaler = @sqrt(2.0 / @as(f64, @floatFromInt(input_size)));
            const hidden_layer_scaler = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_size)));
            const output_layer_scaler = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_size)));

            // init first layer weights and biases
            for (0..hidden_size) |i| {
                for (0..input_size) |j| {
                    self.first_layer_weights[i][j] = (rand.float(f64) * 2.0 - 1.0) * first_layer_scaler;
                }
            }

            // init hidden
            if (hidden_layers > 1) {
                for (0..hidden_layers - 1) |i| {
                    for (0..hidden_size) |j| {
                        for (0..hidden_size) |k| {
                            self.hidden_weights[i][j][k] = (rand.float(f64) * 2.0 - 1.0) * hidden_layer_scaler;
                        }
                    }
                }
            }

            // output
            for (0..hidden_layers) |i| {
                for (0..hidden_size) |j| {
                    self.biases[i][j] = 0;
                }
            }

            for (0..hidden_size) |i| {
                self.output_weights[i] = (rand.float(f64) * 2.0 - 1.0) * output_layer_scaler;
            }
            
            return self;
        }

        // Vector dot product function for SIMD
        inline fn vectorDotProduct(a: []const f64, b: []const f64, len: usize) f64 {
            const zeros: Vec = @splat(0);
            var sum_vec: Vec = zeros;
            var i: usize = 0;

            // 4 at a time
            while (i + 4 <= len) : (i += 4){
                const va: Vec = a[i..][0..4].*;
                const vb: Vec = b[i..][0..4].*;
                sum_vec += va * vb;
            }

            var final_sum: f64 = 0;
            for (0..4) |j| {
                final_sum += sum_vec[j];
            }

            while (i < len) : (i += 1) {
                final_sum += a[i] * b[i];
            }

            return final_sum;

        }

        pub fn predict(self: *const Self, input: *const [input_size]f64) f64 {
            var layer_outputs: [hidden_layers][hidden_size]f64 = undefined;

            // forward pass first layer
            for (0..hidden_size) |j| {
                layer_outputs[0][j] = ReLU(
                    self.biases[0][j] +
                    vectorDotProduct(&self.first_layer_weights[j], input, input_size)
                );
            }

            if (hidden_layers > 1) {
                inline for (1..hidden_layers) |layer| {
                    for (0..hidden_size) |j| {
                        layer_outputs[layer][j] = ReLU(
                            self.biases[layer][j] +
                            vectorDotProduct(&self.hidden_weights[layer - 1][j], &layer_outputs[layer - 1], hidden_size)
                        );
                    }
                }
            }


            // output layer
            return self.bias_output + vectorDotProduct(&self.output_weights, &layer_outputs[hidden_layers - 1], hidden_size);
        }

        pub fn train(self: *Self, input: *const [input_size]f64, target: f64) void {
            var layer_outputs: [hidden_layers][hidden_size]f64 = undefined;

            // forward pass first layer
            for (0..hidden_size) |j| {
                layer_outputs[0][j] = ReLU(
                    self.biases[0][j] +
                    vectorDotProduct(&self.first_layer_weights[j], input, input_size)
                );
            }

            if (hidden_layers > 1) {
                inline for (1..hidden_layers) |layer| {
                    for (0..hidden_size) |j| {
                        layer_outputs[layer][j] = ReLU(
                            self.biases[layer][j] +
                            vectorDotProduct(&self.hidden_weights[layer - 1][j], &layer_outputs[layer - 1], hidden_size)
                        );
                    }
                }
            }


            // output layer
            const output = self.bias_output + vectorDotProduct(&self.output_weights, &layer_outputs[hidden_layers - 1], hidden_size);

            // backwards
            const output_error = mseDeriv(target, output);
            var deltas: [hidden_size]f64 = undefined;

            // Update output weights with SIMD
            inline for (0..hidden_vec_count) |vec_idx| {
                const start_idx = vec_idx * vec_size;
                const end_idx = @min(start_idx + vec_size, hidden_size);
                const vec_len = end_idx - start_idx;
               
                if (vec_len == vec_size) {
                    var weight_vec: Vec = self.output_weights[start_idx..][0..4].*;
                    const output_vec: Vec = layer_outputs[hidden_layers - 1][start_idx..][0..4].*;
                    const gradient_vec = output_vec * @as(Vec, @splat(self.learning_rate * output_error));
                    weight_vec -= gradient_vec;
                    self.output_weights[start_idx..][0..4].* = weight_vec;
                } else {
                    for (start_idx..end_idx) |j| {
                        const gradient = self.learning_rate * output_error * layer_outputs[hidden_layers - 1][j];
                        self.output_weights[j] -= gradient;
                    }
                }
            }
            self.bias_output -= self.learning_rate * output_error;
            

            // back prop
            var layer: usize = hidden_layers - 1;
            while (layer > 0) : (layer -= 1) {
                for (0..hidden_size) |j| {
                    var err: f64 = 0;
                    if (layer == hidden_layers - 1) {
                        err = output_error * self.output_weights[j];
                    } else {
                        err = vectorDotProduct(
                            &deltas,
                            &self.hidden_weights[layer][j],
                            hidden_size
                        );
                    }
                    deltas[j] = err * ReLUDeriv(layer_outputs[layer][j]);
                   
                    if (layer == 1) {
                        for (0..input_size) |k| {
                            const gradient = self.learning_rate * deltas[j] * input[k];
                            self.first_layer_weights[j][k] -= gradient;
                        }
                    } else {
                        for (0..hidden_size) |k| {
                            const gradient = self.learning_rate * deltas[j] * layer_outputs[layer - 1][k];
                            self.hidden_weights[layer - 2][j][k] -= gradient;
                        }
                    }
                    self.biases[layer - 1][j] -= self.learning_rate * deltas[j];
                }
            }
        }
    };
}

inline fn ReLU(x: f64) f64 {
    return if (x > 0) x else 0;
}

inline fn ReLUDeriv(x: f64) f64 {
    return if (x > 0) 1 else 0;
}

inline fn mseDeriv(y: f64, y_pred: f64) f64 {
    return y_pred - y;
}

pub fn main() !void {
    const start = std.time.milliTimestamp();

    // Network configuration
    const input_size = 5;
    const hidden_size = 4;
    const hidden_layers = 2;
    const learning_rate = 0.02;
    const epochs = 1000;

    const Network = NeuralNetwork(input_size, hidden_size, hidden_layers);
    var nn = Network.init(learning_rate);

    const data_length = 100;
    var sin_data: [data_length]f64 = undefined;
    for (0..data_length) |i| {
        const x = @as(f64, @floatFromInt(i)) * 0.1; // * 0.1 helps with accuracy. Not sure why this one is sensetive
        sin_data[i] = std.math.sin(x);
    }

    // training
    for (0..epochs) |_| {
        for (input_size..data_length) |i| {
            var input: [input_size]f64 = undefined;
            for (0..input_size) |j| {
                input[j] = sin_data[i - input_size + j];
            }
            const target = sin_data[i];
            nn.train(&input, target);
        }
    }

    // prediction
    var last_input: [input_size]f64 = undefined;
    for (0..input_size) |j| {
        last_input[j] = sin_data[data_length - input_size + j];
    }

    const prediction = nn.predict(&last_input);
    const x_final = @as(f64, @floatFromInt(data_length)) * 0.1;
    const actual = std.math.sin(x_final);
    const err = @abs(prediction - actual);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("Prediction: {d:.6}\n", .{prediction});
    try stdout.print("Actual: {d:.6}\n", .{actual});
    try stdout.print("Error: {d:.6}\n", .{err});
    
    const end = std.time.milliTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end - start));
    const elapsed_seconds = elapsed_ms / 1000.0;
    try stdout.print("Elapsed time: {d:.6} seconds\n", .{elapsed_seconds});
}

// let's see if it improves
// pretty similar.