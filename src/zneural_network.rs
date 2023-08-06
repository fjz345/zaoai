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
    0.5 * error * error
}

fn node_cost_d(output_activation: f32, expected_activation: f32) -> f32 {
    (output_activation - expected_activation)
}

#[derive(Clone)]
pub struct Layer {
    num_in_nodes: usize,
    num_out_nodes: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    weights_cost_grads: Vec<Vec<f32>>,
    biases_cost_grads: Vec<f32>,
}

#[derive(Clone)]
pub struct LayerLearnData {
    inputs: Vec<f32>,
    weighted_inputs: Vec<f32>,
    activation_values: Vec<f32>,
    //"node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    node_values: Vec<f32>,
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

    fn calculate_outputs(&mut self, activation_inputs: &[f32]) -> Vec<f32> {
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

    fn calculate_outputs_2(
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

    fn apply_cost_gradient(&mut self, learn_rate: f32) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    fn update_cost_gradients(&mut self, learn_data: LayerLearnData) {
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

    fn clear_cost_gradient(&mut self) {
        for node in 0..self.num_out_nodes {
            self.biases_cost_grads[node] = 0.0;

            for weight in 0..self.num_in_nodes {
                self.weights_cost_grads[node][weight] = 0.0;
            }
        }
    }

    fn calculate_output_layer_node_cost_values(
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

    fn calculate_hidden_layer_node_cost_values(
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

impl LayerLearnData {
    fn new(layer: &Layer) -> LayerLearnData {
        LayerLearnData {
            inputs: vec![0.0; layer.num_in_nodes],
            weighted_inputs: vec![0.0; layer.num_out_nodes],
            activation_values: vec![0.0; layer.num_out_nodes],
            node_values: vec![0.0; layer.num_out_nodes],
        }
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
    layer_learn_data: Vec<LayerLearnData>,
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

        let mut layer_learn_data: Vec<LayerLearnData> = Vec::new();
        for i in 0..layers.len() {
            let layer: &Layer = &layers[i];
            layer_learn_data.push(LayerLearnData::new(&layer));
        }

        NeuralNetwork {
            graph_structure,
            layers,
            last_results,
            layer_learn_data,
        }
    }

    pub fn GetLayers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn learn_batch(&mut self, batch_data: &[DataPoint], learn_rate: f32, print: Option<bool>) {
        if (batch_data.len() <= 0) {
            panic!("LearnEpoch DataPoints length was 0");
        }

        let print_enabled = print == Some(true);

        for data_point in batch_data {
            self.update_all_cost_gradients(data_point);
        }
        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (batch_data.len() as f32));
        self.clear_all_cost_gradients();

        // Print cost
        if (print_enabled) {
            let cost = self.calculate_cost(&batch_data[..]);
            println!("Cost: {}", cost);
        }
    }

    pub fn learn_epoch(
        &mut self,
        epoch_index: usize,
        training_data: &[DataPoint],
        batch_size: usize,
        learn_rate: f32,
        print: Option<bool>,
    ) {
        let num_batches = training_data.len() / batch_size;
        let last_batch_size = training_data.len() % batch_size;

        let print_enabled = print == Some(true);

        let mut cur_index: usize = 0;
        let mut batch_step = batch_size;
        if batch_step > training_data.len() {
            batch_step = training_data.len();
        }

        for i in 0..num_batches {
            let epoch_data = &training_data[cur_index..(cur_index + batch_step)];

            if print_enabled {
                println!(
                    "Training... @{} #[{}/{}] (#{} - #{})",
                    epoch_index + 1,
                    i + 1,
                    num_batches,
                    cur_index,
                    cur_index + batch_step,
                );
            }
            self.learn_batch(epoch_data, learn_rate, print);

            cur_index += batch_step;
        }

        if (last_batch_size >= 1) {
            batch_step = last_batch_size;
            // Last epoch
            let epoch_data = &training_data[cur_index..(cur_index + batch_step)];

            if print_enabled {
                println!(
                    "Training... @{} #[{}/{}] (#{} - #{})",
                    epoch_index + 1,
                    num_batches,
                    num_batches,
                    cur_index,
                    cur_index + batch_step,
                );
            }
            self.learn_batch(epoch_data, learn_rate, print);
        }
    }

    pub fn learn(
        &mut self,
        training_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
        print: Option<bool>,
    ) {
        assert!(learn_rate > 0.0);
        assert!(training_data.len() > 0);
        assert!(batch_size > 0);

        let print_enabled = print == Some(true);

        for e in 0..num_epochs {
            if print_enabled {
                println!(
                    "Training...Learn Epoch Started [@{}/@{}]",
                    e + 1,
                    num_epochs
                );
            }

            self.learn_epoch(e, &training_data, batch_size, learn_rate, print);
        }

        if print_enabled {
            println!("Training...Complete! [@{} Epochs]", num_epochs);
        }
    }

    pub fn learn_slow(
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
        let mut num_correct = 0;

        for i in 0..test_data.len() {
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
            num_datapoints: test_data.len() as i32,
            num_correct: num_correct,
            accuracy: (num_correct as f32) / (test_data.len() as f32),
            cost: avg_cost,
        };

        self.last_results = test_result;
        test_result
    }

    // Backpropegation
    fn update_all_cost_gradients(&mut self, data_point: &DataPoint) {
        let mut current_inputs = data_point.inputs.to_vec();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current_inputs =
                layer.calculate_outputs_2(&mut self.layer_learn_data[i], &mut current_inputs);
        }

        // Update for ouput layer
        let layer_len = self.layers.len();
        let mut output_layer = &mut self.layers[layer_len - 1];
        let mut learn_data_output: &mut LayerLearnData = &mut self.layer_learn_data[layer_len - 1];

        output_layer.calculate_output_layer_node_cost_values(
            &mut learn_data_output,
            &data_point.expected_outputs[..],
        );
        output_layer.update_cost_gradients(learn_data_output.clone());

        // Update for hidden layers
        for i in (0..(self.layers.len() - 1)).rev() {
            let hidden_layer: &Layer = &self.layers[i];

            let learn_data_hidden_prev_values = self.layer_learn_data[i + 1].node_values.clone();
            let mut learn_data_hidden: &mut LayerLearnData = &mut self.layer_learn_data[i];

            let prev_layer = &self.layers[i + 1];
            hidden_layer.calculate_hidden_layer_node_cost_values(
                &mut learn_data_hidden,
                prev_layer,
                &learn_data_hidden_prev_values,
            );

            let mut_hidden_layer = &mut self.layers[i];
            mut_hidden_layer.update_cost_gradients(learn_data_hidden.clone());
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
        let mut current_inputs = inputs.to_vec();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current_inputs = layer.calculate_outputs(&current_inputs);
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
