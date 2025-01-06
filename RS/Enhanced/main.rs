use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Instant};
use tokio::task;

#[derive(Debug)]
struct NeuralNetwork {
    input_size: i32,
    hidden_size: i32,
    output_size: i32,
    num_hidden_layers: i32,
    learning_rate: f64,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    bias_output: f64,
}

impl NeuralNetwork {
    fn new(
        input_size: i32,
        hidden_size: i32,
        output_size: i32,
        num_hidden_layers: i32,
        learning_rate: f64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let mut weights = Vec::with_capacity((num_hidden_layers + 1) as usize);
        let mut biases = Vec::with_capacity((num_hidden_layers + 1) as usize);

        // Input to first hidden layer
        weights.push((0..hidden_size).map(|_| {
            (0..input_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
        }).collect());
        biases.push((0..hidden_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect());

        // Hidden layers
        for _ in 1..num_hidden_layers {
            weights.push((0..hidden_size).map(|_| {
                (0..hidden_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
            }).collect());
            biases.push((0..hidden_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect());
        }

        // Last hidden layer to output
        weights.push((0..output_size).map(|_| {
            (0..hidden_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
        }).collect());
        biases.push((0..output_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect());

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            num_hidden_layers,
            learning_rate,
            weights,
            biases,
            bias_output: rng.gen::<f64>() * 2.0 - 1.0,
        }
    }

    fn predict(&self, input: &[f64]) -> f64 {
        let mut current = input.to_vec();
        let mut next_layer = vec![0.0; self.hidden_size as usize];
        // Forward pass through hidden layers
        for layer in 0..self.num_hidden_layers as usize {
            for j in 0..self.hidden_size as usize {
                let sum: f64 = self.biases[layer][j] + current.iter().enumerate().map(|(k, &x)| x * self.weights[layer][j][k]).sum::<f64>();
                next_layer[j] = Self::relu(sum);
            }
            current = next_layer.clone();
        }
        // Output layer
        let output: f64 = self.bias_output + current.iter().enumerate().map(|(j, &x)| x * self.weights[self.weights.len() - 1][0][j]).sum::<f64>();
        output
    }

    // async function for training batches
    async fn train_batch(&mut self, batch: Vec<(Vec<f64>, f64)>) {
        for (input, target) in batch {
            self.train(&input, target).await;
        }
    }

    // function to train Neural Network
    async fn train(&mut self, input: &[f64], target: f64) {
        let mut layer_outputs = Vec::new();
        layer_outputs.push(input.to_vec());

        // Forward pass
        let mut current = input.to_vec();
        for layer in 0..self.num_hidden_layers as usize {
            let next_layer = (0..self.hidden_size as usize).map(|j| {
                let sum: f64 = self.biases[layer][j] + current.iter().enumerate().map(|(k, &x)| x * self.weights[layer][j][k]).sum::<f64>();
                Self::relu(sum)
            }).collect::<Vec<f64>>();
            layer_outputs.push(next_layer.clone());
            current = next_layer;
        }

        //output
        let output = self.bias_output + current.iter().enumerate().map(|(j, &x)| x * self.weights[self.weights.len() - 1][0][j]).sum::<f64>();
       
        // back prop
        let output_error = Self::mse_deriv(target, output);

        // update weights and biases RAH
        for j in 0..self.hidden_size as usize {
            let last_layer = self.weights.len() - 1;
            self.weights[last_layer][0][j] -= self.learning_rate * output_error * layer_outputs[layer_outputs.len() - 1][j];
        }
        self.bias_output -= self.learning_rate * output_error;

        let mut deltas = vec![0.0; self.hidden_size as usize];
        for layer in (0..self.num_hidden_layers as usize).rev() {
            for j in 0..self.hidden_size as usize {
                let error = if layer == (self.num_hidden_layers - 1) as usize {
                    output_error * self.weights[layer + 1][0][j]
                } else {
                    deltas.iter().enumerate().map(|(k, &delta)| delta * self.weights[layer + 1][k][j]).sum()
                };
                deltas[j] = error * Self::relu_deriv(layer_outputs[layer + 1][j]);

                // weights and biases
                for k in 0..layer_outputs[layer].len() {
                    self.weights[layer][j][k] -= self.learning_rate * deltas[j] * layer_outputs[layer][k];
                }
                self.biases[layer][j] -= self.learning_rate * deltas[j];
            }
        }
    }

    fn relu(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn relu_deriv(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn mse_deriv(y: f64, y_pred: f64) -> f64 {
        y_pred - y
    }
}

// needed for the training loop
impl Clone for NeuralNetwork {
    fn clone(&self) -> Self {
        NeuralNetwork {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            output_size: self.output_size,
            num_hidden_layers: self.num_hidden_layers,
            learning_rate: self.learning_rate,
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            bias_output: self.bias_output,
        }
    }
}

#[tokio::main]
async fn main() {
    let now = Instant::now();

    let input_size = 5;
    let hidden_size = 4;
    let output_size = 1;
    let num_hidden_layers = 2;
    let learning_rate = 0.02;
    let epochs = 1000;
    let batch_size = 10;

    let nn = NeuralNetwork::new(input_size, hidden_size, output_size, num_hidden_layers, learning_rate);
    // println!("{:?}", nn);

    let data_length = 100;
    let sin_data: Vec<f64> = (0..data_length).map(|i| (i as f64).sin()).collect();

    for _ in 0..epochs {
        let mut batch = Vec::new();

        for i in (input_size as usize)..data_length {
            let input = sin_data[i - input_size as usize..i].to_vec();
            let target = sin_data[i];
            batch.push((input, target));
            
            if batch.len() >= batch_size {
                let mut nn_clone = nn.clone();
                let batch_clone = batch.clone();
                task::spawn(async move {
                    nn_clone.train_batch(batch_clone).await;
                });
                batch.clear();
            }
        }
    }

    let last_input = &sin_data[data_length - input_size as usize..data_length];
    let prediction = nn.predict(last_input);
    let actual = (data_length as f64).sin();

    println!("Prediction: {:.6}", prediction);
    println!("Actual: {:.6}", actual);
    println!("Error: {:.6}", (prediction - actual).abs());

    let elapsed_time = now.elapsed();
    println!("Elapsed time {} seconds", elapsed_time.as_secs_f64());
    // println!("{:?}", nn);
}
// run time for no enhancements. meh. adding batch training. Remember it was 1 second
// 1 second to 0.35. Nice