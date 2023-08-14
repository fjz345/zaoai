use std::thread::JoinHandle;

use crate::layer::*;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha;
use symphonia::core::util::clamp;

use super::datapoint::DataPoint;

#[derive(Clone)]
pub struct TrainingSession {
    nn: Option<NeuralNetwork>,
    state: TrainingState,
    num_epochs: usize,
    batch_size: usize,
    learn_rate: f32,
    training_data: Vec<DataPoint>,
}

impl TrainingSession {
    pub fn new(
        nn: &NeuralNetwork,
        training_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
    ) -> Self {
        Self {
            nn: Some(nn.clone()),
            state: TrainingState::Idle,
            num_epochs: num_epochs,
            batch_size: batch_size,
            learn_rate: learn_rate,
            training_data: training_data.to_vec(),
        }
    }

    pub fn set_training_data(&mut self, in_data: &[DataPoint]) {
        self.training_data = in_data.to_vec();
    }

    pub fn ready(&self) -> bool {
        self.nn.is_some()
            && self.training_data.len() > 0
            && self.num_epochs > 0
            && self.batch_size > 0
            && self.learn_rate > 0.0
    }

    // Spawns a new thread and begins learning the supplied NN
    pub fn begin(&self) -> JoinHandle<()> {
        let nn_option = self.nn.clone();
        let training_data = self.training_data.clone();
        let num_epochs = self.num_epochs;
        let batch_size = self.batch_size;
        let learn_rate = self.learn_rate;

        let training_thread = std::thread::spawn(move || {
            if nn_option.is_some() {
                let mut nn: NeuralNetwork = nn_option.unwrap().clone();
                nn.learn(
                    &training_data[..],
                    num_epochs,
                    batch_size,
                    learn_rate,
                    Some(false),
                );
            }
        });

        return training_thread;
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

#[derive(Clone)]
pub struct GraphStructure {
    pub input_nodes: usize,
    pub hidden_layers: Vec<usize>, // contais nodes
    pub output_nodes: usize,
    pub use_softmax_output: bool,
}

impl GraphStructure {
    pub fn new(args: &[usize], use_softmax_output: bool) -> GraphStructure {
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
            use_softmax_output: use_softmax_output,
        }
    }

    pub fn validate(&self) -> bool {
        let mut is_valid = true;

        if self.input_nodes < 1 {
            is_valid = false;
        } else if self.output_nodes < 1 {
            is_valid = false;
        }

        is_valid
    }

    fn to_string(&self) -> String {
        let mut result_string: String = String::new();
        let mut layer_sizes: Vec<usize> = Vec::new();

        layer_sizes.push(self.input_nodes);
        for hidden_layer in &self.hidden_layers[..] {
            layer_sizes.push(*hidden_layer);
        }

        layer_sizes.push(self.output_nodes);

        for (i, layer) in layer_sizes.iter().enumerate() {
            if (i >= 1) {
                result_string += ", ";
            }

            result_string += layer.to_string().as_str();
        }

        result_string
    }

    fn print(&self) {
        // Print
        println!("{}", self.to_string());
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TestResults {
    pub num_datapoints: i32,
    pub num_correct: i32,
    pub accuracy: f32,
    pub cost: f32,
}

#[derive(Clone)]
pub enum TrainingState {
    Idle,
    Training,
    Finish,
    Abort,
}

#[derive(Clone)]
pub struct NeuralNetwork {
    pub graph_structure: GraphStructure,
    pub layers: Vec<Layer>,
    pub last_test_results: TestResults,
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
            last_test_results: last_results,
            layer_learn_data,
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn learn_batch(&mut self, batch_data: &[DataPoint], learn_rate: f32, print: Option<bool>) {
        if (batch_data.len() <= 0) {
            panic!("DataPoints length was 0");
        }

        let print_enabled = print == Some(true);

        for datapoint in batch_data {
            self.update_all_cost_gradients(datapoint);
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

    // Not recommended for use
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

        self.last_test_results = test_result;
        test_result
    }

    // Backpropegation
    fn update_all_cost_gradients(&mut self, datapoint: &DataPoint) {
        let mut current_inputs = datapoint.inputs.to_vec();
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
            &datapoint.expected_outputs[..],
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

        // TODO: need to fix backpropegation
        // if self.graph_structure.use_softmax_output {
        //     current_inputs = softmax(&current_inputs);
        // }

        current_inputs
    }

    pub fn calculate_outputs_softmax(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut current_inputs = self.calculate_outputs(inputs);
        softmax(&current_inputs)
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

    fn calculate_cost_datapoint(&mut self, datapoint: DataPoint) -> f32 {
        let mut cost: f32 = 0.0;

        // Calculate the output of the neural network
        let datapoint_output = self.calculate_outputs(&datapoint.inputs[..]);

        // Calculate cost by comparing difference between output and expected output
        let output_layer = self.layers.last().unwrap();
        for output_node in 0..output_layer.num_out_nodes {
            cost += node_cost(
                datapoint_output[output_node],
                datapoint.expected_outputs[output_node],
            );
        }

        cost
    }

    pub fn calculate_cost(&mut self, data: &[DataPoint]) -> f32 {
        if data.len() <= 0 {
            panic!("Input data was len: {}", data.len());
        }

        let mut cost: f32 = 0.0;
        for datapoint in &data[..] {
            cost += self.calculate_cost_datapoint(*datapoint);
        }

        cost / (data.len() as f32)
    }

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

    pub fn to_string(&self) -> String {
        let print_string: String = format!(
            "\
        Graph Structure: {}\n\
        Last Test Results: {:#?}\n",
            self.graph_structure.to_string(),
            self.last_test_results
        );

        print_string
    }

    pub fn print(&self) {
        print!("----------NEURAL NETWORK----------\n");
        print!("{}", self.to_string());
        print!("----------------------------------\n");
    }
}
