use rand::prelude::*;
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

fn activation_function(in_value: f32) -> f32 {
    //relu(in_value)
    sigmoid(in_value)
}

fn activation_function_d(in_value: f32) -> f32 {
    //relu_d(in_value)
    sigmoid_d(in_value)
}

fn node_cost(output_activation: f32, expected_activation: f32) -> f32 {
    let error = output_activation - expected_activation;
    error * error
}

fn node_cost_d(output_activation: f32, expected_activation: f32) -> f32 {
    2.0 * (output_activation - expected_activation)
}

#[derive(Clone)]
pub struct Layer {
    num_in_nodes: usize,
    num_out_nodes: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    weights_cost_grads: Vec<Vec<f32>>,
    biases_cost_grads: Vec<f32>,
    activation_values: Vec<f32>,
    weighted_inputs: Vec<f32>,
}

impl Layer {
    fn new(num_in_nodes: usize, num_out_nodes: usize) -> Layer {
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

        // Initialize weights & biases
        for i in 0..num_out_nodes {
            let rand_bias: f32 = random();
            biases[i] = rand_bias;

            for j in 0..num_in_nodes {
                let rand_weight: f32 = random();
                weights[i][j] = rand_weight;
            }
        }

        let mut activation_values: Vec<f32> = vec![0.0; num_in_nodes];

        let mut weighted_inputs: Vec<f32> = vec![0.0; num_out_nodes];

        Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
            activation_values,
            weighted_inputs,
        }
    }

    fn calculate_outputs(&mut self, activation_inputs: &[f32]) -> Vec<f32> {
        let mut activation_outputs = vec![0.0; self.num_out_nodes];

        assert_eq!(
            activation_inputs.len(),
            self.num_in_nodes,
            "Num Inputs: {}, NN Num Inputs {}. Maybe data missmatch with output size?",
            activation_inputs.len(),
            self.num_in_nodes
        );

        self.activation_values = activation_inputs.to_vec();

        for (output_node, output) in activation_outputs.iter_mut().enumerate() {
            // First apply bias for the output node
            self.weighted_inputs[output_node] = 0.0;

            // Then apply input node weights
            for j in 0..activation_inputs.len() {
                self.weighted_inputs[output_node] +=
                    activation_inputs[j] * self.weights[output_node][j];
            }

            // First apply bias for the output node
            self.weighted_inputs[output_node] += self.biases[output_node];

            *output = self.weighted_inputs[output_node];
        }

        activation_outputs
    }

    fn apply_cost_gradient(&mut self, learn_rate: f32) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    fn update_cost_gradients(&mut self, node_cost_values: &[f32]) {
        for node_out in 0..self.num_out_nodes {
            // Weight costs
            for node_in in 0..self.num_in_nodes {
                let derivative_cost_weight =
                    self.activation_values[node_in] * node_cost_values[node_out];
                self.weights_cost_grads[node_out][node_in] += derivative_cost_weight;
            }

            // Bias costs
            let derivative_cost_bias = 1.0 * node_cost_values[node_out];
            self.biases_cost_grads[node_out] += derivative_cost_bias;
        }
    }

    fn clear_cost_gradient(&mut self) {
        for node in 0..self.num_out_nodes {
            self.biases_cost_grads[node] = 0.0;

            for weight in 0..self.num_in_nodes {
                self.weights_cost_grads[node][weight] = 0.0;
            }
        }
    }

    fn calculate_output_layer_node_cost_values(&self, expected_outputs: &[f32]) -> Vec<f32> {
        let mut node_cost_values: Vec<f32> = vec![0.0; expected_outputs.len()];

        for i in 0..node_cost_values.len() {
            let dcost = node_cost_d(self.activation_values[i], expected_outputs[i]);
            let dactivation = activation_function_d(self.weighted_inputs[i]);
            node_cost_values[i] = dactivation * dcost;
        }

        node_cost_values
    }

    fn calculate_hidden_layer_node_cost_values(
        &self,
        prev_layer: &Layer,
        prev_node_cost_values: &[f32],
    ) -> Vec<f32> {
        let mut new_node_cost_values: Vec<f32> = vec![0.0; self.num_out_nodes];

        for new_node_index in 0..new_node_cost_values.len() {
            let mut new_node_cost_value: f32 = 0.0;
            for prev_node_index in 0..prev_node_cost_values.len() {
                let weighted_input_d = prev_layer.weights[prev_node_index][new_node_index];
                new_node_cost_value += weighted_input_d * prev_node_cost_values[prev_node_index];
            }
            new_node_cost_value *= activation_function_d(self.weighted_inputs[new_node_index]);
            new_node_cost_values[new_node_index] = new_node_cost_value;
        }

        new_node_cost_values
    }
}

pub struct GraphStructure {
    pub input_nodes: usize,
    pub hidden_layers: Vec<usize>, // contais nodes
    pub output_nodes: usize,
}

impl GraphStructure {
    pub fn new(args: &[usize]) -> GraphStructure {
        if args.len() < 2 {
            // Format args to string
            let mut output_string: String = "".to_owned();
            for arg in args {
                output_string.push_str(arg.to_string().as_str());
            }
            panic!(
                "GraphStructure had no input and output layer, provided: {}",
                output_string
            );
        }

        let input_nodes: usize = args[0];

        let mut hidden_nodes: Vec<usize> = Vec::new();
        for arg in &args[1..(args.len() - 1)] {
            hidden_nodes.push(*arg);
        }

        let output_nodes: usize = args.last().unwrap().clone();

        GraphStructure {
            input_nodes,
            hidden_layers: hidden_nodes.clone(),
            output_nodes,
        }
    }

    fn validate(&self) -> bool {
        let mut is_valid = true;

        if self.input_nodes < 1 {
            is_valid = false;
        } else if self.output_nodes < 1 {
            is_valid = false;
        } else if self.input_nodes != self.output_nodes {
            is_valid = false;
        }

        is_valid
    }

    fn print(&self) {
        // Create a vec with the sizes (unwrap the hidden layer and combine)
        let mut layer_sizes: Vec<usize> = Vec::new();

        layer_sizes.push(self.input_nodes);
        for hidden_layer in &self.hidden_layers[..] {
            layer_sizes.push(*hidden_layer);
        }

        layer_sizes.push(self.output_nodes);

        // Print
        println!("{:?}", layer_sizes);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DataPoint {
    pub inputs: [f32; 2],
    pub expected_outputs: [f32; 2],
}

#[derive(Copy, Clone, Debug)]
pub struct TestResults {
    pub num_datapoints: i32,
    pub num_correct: i32,
    pub accuracy: f32,
    pub cost: f32,
}

pub struct NeuralNetwork {
    pub graph_structure: GraphStructure,
    pub layers: Vec<Layer>,
    pub last_results: TestResults,
}

impl NeuralNetwork {
    pub fn new(graph_structure: GraphStructure) -> NeuralNetwork {
        let mut layers: Vec<Layer> = Vec::new();
        let mut prev_out_size = graph_structure.input_nodes;

        // Input nodes are not layers in the neural network.

        // Create Hidden layers
        for i in &graph_structure.hidden_layers[..] {
            layers.push(Layer::new(prev_out_size, *i));
            prev_out_size = *i;
        }

        // Create Output layer
        layers.push(Layer::new(prev_out_size, graph_structure.output_nodes));

        let last_results = TestResults {
            num_datapoints: 0,
            num_correct: 0,
            accuracy: 0.0,
            cost: 0.0,
        };
        NeuralNetwork {
            graph_structure,
            layers,
            last_results,
        }
    }

    pub fn GetLayers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn learn_epoch(&mut self, epoch_data: &[DataPoint], learn_rate: f32, print: Option<bool>) {
        if (epoch_data.len() <= 0) {
            panic!("LearnEpoch DataPoints length was 0");
        }

        let print_enabled = print == Some(true);

        for data_point in epoch_data {
            self.update_all_cost_gradients(data_point);
        }

        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (epoch_data.len() as f32));

        self.clear_all_cost_gradients();

        // Print cost
        if (print_enabled) {
            let cost = self.calculate_cost(&epoch_data[..]);
            println!("Cost: {}", cost);
        }
    }

    pub fn learn(
        &mut self,
        training_data: &[DataPoint],
        epochs_size: usize,
        learn_rate: f32,
        print: Option<bool>,
    ) {
        if (training_data.len() <= 0) {
            panic!("Learn DataPoints length was 0");
        }

        if (epochs_size <= 0) {
            panic!("Learn Num Epochs was 0");
        }

        let print_enabled = print == Some(true);

        let num_epochs = training_data.len() / epochs_size;
        let last_epoch_size = training_data.len() % epochs_size;

        if print_enabled {
            println!(
                "Training...Learn Started [{}, {}]",
                training_data.len(),
                num_epochs
            );
        }

        let mut cur_index: usize = 0;
        let mut epoch_step = epochs_size;
        if epoch_step > training_data.len() {
            epoch_step = training_data.len();
        }

        for i in 0..num_epochs {
            let epoch_data = &training_data[cur_index..(cur_index + epoch_step)];

            if print_enabled {
                println!(
                    "Training...Epoch [{}/{}] @{} (#{} - #{})",
                    i,
                    num_epochs,
                    epoch_step,
                    cur_index,
                    cur_index + epoch_step
                );
            }
            self.learn_epoch(epoch_data, learn_rate, print);

            cur_index += epoch_step;
        }

        if (last_epoch_size >= 1) {
            epoch_step = last_epoch_size;
            // Last epoch
            let epoch_data = &training_data[cur_index..(cur_index + epoch_step)];

            if print_enabled {
                println!(
                    "Training...Epoch [{}/{}] @{} (#{} - #{})",
                    num_epochs,
                    num_epochs,
                    epoch_step,
                    cur_index,
                    cur_index + epoch_step
                );
            }
            self.learn_epoch(epoch_data, learn_rate, print);
        }

        if print_enabled {
            println!("Training...Complete! [{}/{}]", num_epochs, num_epochs);
        }
    }

    pub fn learn_old(
        &mut self,
        training_data: &[DataPoint],
        mut num_epochs: usize,
        learn_rate: f32,
        print: Option<bool>,
    ) {
        if (training_data.len() <= 0) {
            panic!("Learn DataPoints length was 0");
        }

        let print_enabled = print == Some(true);

        num_epochs = num_epochs.min((training_data.len() / num_epochs) + 1);

        if print_enabled {
            println!(
                "Training...Learn Started [{}, {}]",
                training_data.len(),
                num_epochs
            );
        }

        let h: f32 = 0.00001;

        let mut cur_index: usize = 0;
        let mut epoch_step = (training_data.len() / num_epochs) + 1;
        if epoch_step > training_data.len() {
            epoch_step = training_data.len();
        }

        for i in 0..num_epochs {
            if (cur_index + epoch_step >= (training_data.len())) {
                break;
            }

            if print_enabled {
                println!(
                    "Training...Epoch [{}/{}] @({} - {})",
                    i,
                    num_epochs,
                    cur_index,
                    cur_index + epoch_step
                );
            }

            let epoch_data = &training_data[cur_index..(cur_index + epoch_step)];

            let original_cost = self.calculate_cost(&epoch_data[..]);

            if print_enabled {
                println!("Cost: {}", original_cost);
            }

            // Calculate cost gradients for layers
            for i in 0..self.layers.len() {
                // Weights
                for in_node in 0..self.layers[i].num_in_nodes {
                    for out_node in 0..self.layers[i].num_out_nodes {
                        self.layers[i].weights[out_node][in_node] += h;
                        let dcost = self.calculate_cost(&epoch_data[..]) - original_cost;
                        self.layers[i].weights[out_node][in_node] -= h;
                        self.layers[i].weights_cost_grads[out_node][in_node] = dcost / h;
                    }
                }

                // Biases
                for bias_index in 0..self.layers[i].biases.len() {
                    self.layers[i].biases[bias_index] += h;
                    let dcost = self.calculate_cost(&epoch_data[..]) - original_cost;
                    self.layers[i].biases[bias_index] -= h;
                    self.layers[i].biases_cost_grads[bias_index] = dcost / h;
                }
            }

            // Adjust weights & biases
            self.apply_all_cost_gradients(learn_rate);
            self.clear_all_cost_gradients();

            cur_index += epoch_step;
        }
    }

    // Returns a TestResult
    pub fn test(&mut self, test_data: &[DataPoint]) -> TestResults {
        let num_datapoints = test_data.len();
        let mut num_correct = 0;

        for i in 0..num_datapoints {
            let mut datapoint = test_data[i];
            let outputs = self.calculate_outputs(&mut datapoint.inputs[..]);
            let result = NeuralNetwork::determine_output_result(&outputs);
            let result_expected =
                NeuralNetwork::determine_output_result(&datapoint.expected_outputs);

            let is_correct = result.0 == result_expected.0;
            if (is_correct) {
                num_correct += 1;
            }
        }

        let avg_cost = self.calculate_cost(test_data);
        let test_result = TestResults {
            num_datapoints: num_datapoints as i32,
            num_correct: num_correct,
            accuracy: (num_correct as f32) / (num_datapoints as f32),
            cost: avg_cost,
        };

        self.last_results = test_result;
        test_result
    }

    fn update_all_cost_gradients(&mut self, data_point: &DataPoint) {
        // Calculate node values (populate layers::activation_values, weighted_inputs)
        let o = self.calculate_outputs(&data_point.inputs);

        // Update for ouput layer
        let layer_len = self.layers.len();
        let mut output_layer = &mut self.layers[layer_len - 1];
        let mut node_cost_values =
            output_layer.calculate_output_layer_node_cost_values(&data_point.expected_outputs[..]);
        output_layer.update_cost_gradients(&node_cost_values);

        // Update for hidden layers
        for i in (0..(self.layers.len() - 1)).rev() {
            let hidden_layer = &self.layers[i];
            let prev_layer = &self.layers[i + 1];
            node_cost_values =
                hidden_layer.calculate_hidden_layer_node_cost_values(prev_layer, &node_cost_values);

            let mut_hidden_layer = &mut self.layers[i];
            mut_hidden_layer.update_cost_gradients(&node_cost_values);
        }
    }

    fn apply_all_cost_gradients(&mut self, learn_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.apply_cost_gradient(learn_rate);
        }
    }

    fn clear_all_cost_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cost_gradient();
        }
    }

    pub fn calculate_outputs(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut current_inputs = vec![0.0; inputs.len()];
        current_inputs.clone_from_slice(inputs);
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current_inputs = layer.calculate_outputs(&mut current_inputs);
        }

        current_inputs
    }

    // returns index of max value, max value
    pub fn determine_output_result(inputs: &[f32]) -> (usize, f32) {
        let mut max = -99999999999.0;
        let mut max_index = 0;
        // Choose the greatest value
        for (i, input) in inputs.iter().enumerate() {
            if (*input > max) {
                max = *input;
                max_index = i;
            }
        }

        (max_index, max)
    }

    fn calculate_cost_data_point(&mut self, data_point: DataPoint) -> f32 {
        let mut cost: f32 = 0.0;

        // Calculate the output of the neural network
        let data_point_output = self.calculate_outputs(&data_point.inputs[..]);

        // Calculate cost by comparing difference between output and expected output
        let output_layer = self.layers.last().unwrap();
        for output_node in 0..output_layer.num_out_nodes {
            cost += node_cost(
                data_point_output[output_node],
                data_point.expected_outputs[output_node],
            );
        }

        cost
    }

    pub fn calculate_cost(&mut self, data: &[DataPoint]) -> f32 {
        if data.len() <= 0 {
            panic!("Input data was len: {}", data.len());
        }

        let mut cost: f32 = 0.0;

        for data_point in &data[..] {
            cost += self.calculate_cost_data_point(*data_point);
        }

        cost / (data.len() as f32)
    }

    // Checks that the input size matches the output size between layers
    pub fn validate(&self) -> bool {
        let mut is_valid: bool = true;

        // Validate Graph Structure
        if !self.graph_structure.validate() {
            is_valid = false;
        }

        // Ensure that the layers input/output numbers match
        let mut prev_out_size: usize = self.graph_structure.input_nodes;
        for layer in &self.layers[..] {
            if (layer.num_in_nodes != prev_out_size) {
                is_valid = false;
                break;
            }

            prev_out_size = layer.num_out_nodes;
        }

        // TODO: validate In_nodes & out_nodes with graph_strucutre values also

        is_valid
    }

    // Prints each layer, showing the number of nodes in each layer
    pub fn print(&self) {
        println!("----------NEURAL NETWORK----------");
        println!("Graph Structure: ");
        self.graph_structure.print();

        // Print Actual layer sizes
        println!("Layer Nodes: ");
        print!("[{:?}", self.graph_structure.input_nodes);
        for layer in &self.layers[..] {
            print!(", ");
            print!("{:?}", layer.num_out_nodes);
        }
        print!("]\n");

        ///////////////////////////////////////////////////

        println!("Last Test Results: ");
        println!("{:#?}", self.last_results);
        println!("----------------------------------");
    }
}
