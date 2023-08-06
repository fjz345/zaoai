use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha;
use symphonia::core::util::clamp;

fn sigmoid(in_value: f32) -> f32 {
    1.0 / (1.0 + (-in_value).exp())
}

fn sigmoid_d(in_value: f32) -> f32 {
    let f = sigmoid(in_value);
    f * (1.0 - f)
}

fn relu(in_value: f32) -> f32 {
    in_value.max(0.0)
}

fn relu_d(in_value: f32) -> f32 {
    !panic!("not implemented");
}

pub fn activation_function(in_value: f32) -> f32 {
    //relu(in_value)
    sigmoid(in_value)
}

pub fn activation_function_d(in_value: f32) -> f32 {
    //relu_d(in_value)
    sigmoid_d(in_value)
}

pub fn node_cost(output_activation: f32, expected_activation: f32) -> f32 {
    let error = output_activation - expected_activation;
    0.5 * error * error
}

pub fn node_cost_d(output_activation: f32, expected_activation: f32) -> f32 {
    (output_activation - expected_activation)
}

#[derive(Clone)]
pub struct Layer {
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub weights_cost_grads: Vec<Vec<f32>>,
    pub biases_cost_grads: Vec<f32>,
}

#[derive(Clone)]
pub struct LayerLearnData {
    pub inputs: Vec<f32>,
    pub weighted_inputs: Vec<f32>,
    pub activation_values: Vec<f32>,
    //"node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub node_values: Vec<f32>,
}

impl Layer {
    pub fn new(num_in_nodes: usize, num_out_nodes: usize) -> Layer {
        // Validate Inputs
        if num_in_nodes <= 0 {
            panic!("NumInNodes must be > 0, got {}", num_in_nodes);
        }
        if num_out_nodes <= 0 {
            panic!("NumOutNodes must be > 0, got {}", num_out_nodes);
        }

        // Allocate memory
        // Bias
        let mut biases: Vec<f32> = Vec::new();
        biases.resize(num_out_nodes, 0.0);
        let mut biases_cost_grads: Vec<f32> = Vec::new();
        biases_cost_grads.resize(num_out_nodes, 0.0);

        // Weight
        let mut weights: Vec<Vec<f32>> = Vec::new();
        weights.reserve(num_out_nodes);
        for i in 0..num_out_nodes {
            weights.push(vec![0.0; num_in_nodes]);
        }
        let mut weights_cost_grads: Vec<Vec<f32>> = Vec::new();
        weights_cost_grads.reserve(num_out_nodes);
        for i in 0..num_out_nodes {
            weights_cost_grads.push(vec![0.0; num_in_nodes]);
        }

        let mut activation_values: Vec<f32> = vec![0.0; num_in_nodes];
        let mut weighted_inputs: Vec<f32> = vec![0.0; num_out_nodes];

        let mut new_layer = Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
        };

        new_layer.init_weights_and_biases(0);

        new_layer
    }

    fn init_weights_and_biases(&mut self, seed: u64) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let min_weight = 0.0;
        let max_weight = 1.0;
        let min_bias = min_weight;
        let max_bias = max_weight;

        // Initialize weights & biases
        for i in 0..self.num_out_nodes {
            let rand_bias: f32 = rng.gen_range(min_weight..max_weight);
            self.biases[i] = rand_bias;

            for j in 0..self.num_in_nodes {
                let rand_weight: f32 = rng.gen_range(min_bias..max_bias);
                self.weights[i][j] = rand_weight;
            }
        }
    }

    pub fn calculate_outputs(&mut self, activation_inputs: &[f32]) -> Vec<f32> {
        let mut weighted_inputs = vec![0.0; self.num_out_nodes];

        assert_eq!(
            activation_inputs.len(),
            self.num_in_nodes,
            "Num Inputs: {}, NN Num Inputs {}. Maybe data missmatch with output size?",
            activation_inputs.len(),
            self.num_in_nodes
        );

        for output_node in 0..self.num_out_nodes {
            let mut weighted_input = self.biases[output_node];
            for input in 0..activation_inputs.len() {
                weighted_input += activation_inputs[input] * self.weights[output_node][input];
            }
            weighted_inputs[output_node] = weighted_input;
        }

        let mut activation_outputs = vec![0.0; self.num_out_nodes];
        for output_node in 0..self.num_out_nodes {
            activation_outputs[output_node] = activation_function(weighted_inputs[output_node]);
        }

        activation_outputs
    }

    pub fn calculate_outputs_2(
        &mut self,
        learn_data: &mut LayerLearnData,
        activation_inputs: &[f32],
    ) -> Vec<f32> {
        assert_eq!(
            activation_inputs.len(),
            self.num_in_nodes,
            "Num Inputs: {}, NN Num Inputs {}. Maybe data missmatch with output size?",
            activation_inputs.len(),
            self.num_in_nodes
        );

        learn_data.inputs.clone_from_slice(activation_inputs);

        for output_node in 0..self.num_out_nodes {
            let mut weighted_input = self.biases[output_node];
            for input_node in 0..self.num_in_nodes {
                weighted_input +=
                    activation_inputs[input_node] * self.weights[output_node][input_node];
            }
            learn_data.weighted_inputs[output_node] = weighted_input;
        }

        for i in 0..learn_data.activation_values.len() {
            learn_data.activation_values[i] = activation_function(learn_data.weighted_inputs[i]);
        }

        learn_data.activation_values.clone()
    }

    pub fn apply_cost_gradient(&mut self, learn_rate: f32) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    pub fn update_cost_gradients(&mut self, learn_data: LayerLearnData) {
        for node_out in 0..self.num_out_nodes {
            // Weight costs
            for node_in in 0..self.num_in_nodes {
                let derivative_cost_weight =
                    learn_data.inputs[node_in] * learn_data.node_values[node_out];
                self.weights_cost_grads[node_out][node_in] += derivative_cost_weight;
            }

            // Bias costs
            let derivative_cost_bias = 1.0 * learn_data.node_values[node_out];
            self.biases_cost_grads[node_out] += derivative_cost_bias;
        }
    }

    pub fn clear_cost_gradient(&mut self) {
        for node in 0..self.num_out_nodes {
            self.biases_cost_grads[node] = 0.0;

            for weight in 0..self.num_in_nodes {
                self.weights_cost_grads[node][weight] = 0.0;
            }
        }
    }

    pub fn calculate_output_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[f32],
    ) {
        for i in 0..learn_data.node_values.len() {
            let dcost = node_cost_d(learn_data.activation_values[i], expected_outputs[i]);
            let dactivation = activation_function_d(learn_data.weighted_inputs[i]);
            learn_data.node_values[i] = dactivation * dcost;
        }
    }

    pub fn calculate_hidden_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        prev_layer: &Layer,
        prev_node_cost_values: &[f32],
    ) {
        for new_node_index in 0..self.num_out_nodes {
            let mut new_node_value: f32 = 0.0;
            for prev_node_index in 0..prev_node_cost_values.len() {
                let weighted_input_d = prev_layer.weights[prev_node_index][new_node_index];
                new_node_value += weighted_input_d * prev_node_cost_values[prev_node_index];
            }
            new_node_value *= activation_function_d(learn_data.weighted_inputs[new_node_index]);
            learn_data.node_values[new_node_index] = new_node_value;
        }
    }
}
