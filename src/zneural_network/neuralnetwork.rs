use crate::layer::*;
use crate::zneural_network::thread::TrainingThreadPayload;
use crate::zneural_network::training::{AIResultMetadata, DatasetUsage, TestResults};

use super::datapoint::DataPoint;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use symphonia::core::util::clamp;

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

#[derive(Clone, Serialize, Deserialize, Debug)]
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

    pub fn validate(&self) -> bool {
        let mut is_valid = true;

        if self.input_nodes < 1 {
            is_valid = false;
        } else if self.output_nodes < 1 {
            is_valid = false;
        }

        is_valid
    }

    pub fn to_string(&self) -> String {
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
        log::info!("{}", self.to_string());
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralNetwork {
    pub graph_structure: GraphStructure,
    pub layers: Vec<Layer>,
    pub last_test_results: TestResults,
    #[serde(skip)]
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

    fn cross_entropy_loss(predicted: &[f32], expected: &[f32]) -> f32 {
        // Small epsilon to avoid log(0)
        let epsilon = 1e-12;

        predicted
            .iter()
            .zip(expected.iter())
            .map(|(p, e)| {
                // Clamp p to avoid log(0)
                let p_clamped = p.max(epsilon).min(1.0 - epsilon);
                -e * p_clamped.ln()
            })
            .sum()
    }

    pub fn learn_batch(
        &mut self,
        batch_data: &[DataPoint],
        learn_rate: f32,
        batch_data_cost: &mut f32,
        batch_data_loss: &mut f32,
        print: Option<bool>,
    ) -> Vec<Vec<f32>> {
        // let new_metadata = AIResultMetadata::new(DatasetUsage::Training);

        if (batch_data.len() <= 0) {
            panic!("DataPoints length was 0");
        }

        let print_enabled = print == Some(true);

        let mut total_loss = 0.0;
        let mut batch_data_outputs = Vec::new();
        for datapoint in batch_data {
            // Todo: make functions forward/backward for simplicity.
            let datapoint_outputs = self.update_all_cost_gradients(datapoint);
            let loss = Self::cross_entropy_loss(&datapoint_outputs, &datapoint.expected_outputs);
            total_loss += loss;

            batch_data_outputs.push(datapoint_outputs);
        }
        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (batch_data.len() as f32));
        self.clear_all_cost_gradients();

        *batch_data_cost = self.calculate_cost(&batch_data[..]);
        *batch_data_loss = total_loss;
        // Print cost
        if (print_enabled) {
            log::info!("Cost: {}", batch_data_cost);
        }

        batch_data_outputs
    }

    pub fn learn_epoch(
        &mut self,
        epoch_index: usize,
        training_data: &[DataPoint],
        batch_size: usize,
        learn_rate: f32,
        print: Option<bool>,
        mut epoch_metadata: Option<&mut AIResultMetadata>,
    ) {
        assert!(!training_data.is_empty());
        assert_eq!(
            self.graph_structure.input_nodes,
            training_data[0].inputs.len()
        );
        assert_eq!(
            self.graph_structure.output_nodes,
            training_data[0].expected_outputs.len()
        );

        let print_enabled = print.unwrap_or(false);
        let mut cur_index = 0;
        let len = training_data.len();

        let mut process_batch =
            |data: &[DataPoint], batch_num: usize, total_batches: usize, cur_index: usize| {
                if print_enabled {
                    log::info!(
                        "Training... @{} #[{}/{}] (#{} - #{})",
                        epoch_index + 1,
                        batch_num + 1,
                        total_batches,
                        cur_index,
                        cur_index + data.len(),
                    );
                }

                let mut batch_data_cost = 0.0;
                let mut batch_data_loss = 0.0;
                let batch_data_outputs = self.learn_batch(
                    data,
                    learn_rate,
                    &mut batch_data_cost,
                    &mut batch_data_loss,
                    print,
                );

                if let Some(metadata) = epoch_metadata.as_mut() {
                    let mut new_metadata = AIResultMetadata::new(
                        DatasetUsage::Training,
                        batch_data_cost as f64,
                        batch_data_loss as f64,
                    );
                    self.learn_batch_metadata(
                        data,
                        &batch_data_outputs,
                        batch_data_cost,
                        &mut new_metadata,
                    );
                    metadata.merge(&new_metadata);
                }
            };

        let num_batches = len / batch_size;
        let last_batch_size = len % batch_size;

        for i in 0..num_batches {
            let batch = &training_data[cur_index..cur_index + batch_size];
            process_batch(batch, i, num_batches, cur_index);
            cur_index += batch_size;
        }

        if last_batch_size > 0 {
            let batch = &training_data[cur_index..];
            process_batch(batch, num_batches, num_batches, cur_index);
        }
    }

    fn learn_batch_metadata(
        &self,
        epoch_data: &[DataPoint],
        epoch_data_outputs: &Vec<Vec<f32>>,
        epoch_data_cost: f32,
        new_metadata: &mut AIResultMetadata,
    ) {
        for (i, data) in epoch_data.iter().enumerate() {
            let datapoint_output = &epoch_data_outputs[i];

            let (determined_index, determined_value) =
                Self::determine_output_result(&datapoint_output[..]);

            let (determined_expected_index, determined_expected_value) =
                Self::determine_output_result(&data.expected_outputs);

            match (determined_index == determined_expected_index, false) {
                (true, false) => {
                    new_metadata.true_positives += 1;
                    new_metadata.positive_instances += 1;
                }
                (false, false) => {
                    new_metadata.false_positives += 1;
                    new_metadata.negative_instances += 1;
                }
                _ => panic!("Fix bug"),
            }
        }

        new_metadata.cost = epoch_data_cost as f64;
    }

    pub fn learn(
        &mut self,
        training_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
        print: Option<bool>,
        tx_training_metadata: Option<&Sender<TrainingThreadPayload>>,
    ) {
        assert!(learn_rate > 0.0);
        assert!(training_data.len() > 0);
        assert!(batch_size > 0);

        let print_enabled = print == Some(true);

        for e in 0..num_epochs {
            if print_enabled {
                log::info!(
                    "Training...Learn Epoch Started [@{}/@{}]",
                    e + 1,
                    num_epochs
                );
            }

            let mut metadata: AIResultMetadata =
                AIResultMetadata::new(DatasetUsage::Training, 0.0, 0.0);
            self.learn_epoch(
                e,
                &training_data,
                batch_size,
                learn_rate,
                print,
                Some(&mut metadata),
            );

            if tx_training_metadata.is_some() {
                let payload = TrainingThreadPayload {
                    payload_index: e,
                    payload_max_index: num_epochs - 1,
                    training_metadata: metadata,
                };
                tx_training_metadata.unwrap().send(payload);
            }
        }

        if print_enabled {
            log::info!("Training...Complete! [@{} Epochs]", num_epochs);
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
            log::info!(
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
                log::info!(
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
                log::info!("Cost: {}", original_cost);
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

    // Backpropegation
    fn update_all_cost_gradients(&mut self, datapoint: &DataPoint) -> Vec<f32> {
        // Doubt it is the bottleneck, but could reuse a buffer instead of getting a new one each time.
        let mut current_inputs = datapoint.inputs.to_vec();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current_inputs = layer
                .calculate_outputs_learn_simd(&mut self.layer_learn_data[i], &mut current_inputs);
        }

        let output_inputs = current_inputs;

        // Update for ouput layer
        let layer_len = self.layers.len();
        let mut output_layer = &mut self.layers[layer_len - 1];
        let mut learn_data_output: &mut LayerLearnData = &mut self.layer_learn_data[layer_len - 1];

        output_layer.calculate_output_layer_node_cost_values(
            &mut learn_data_output,
            &datapoint.expected_outputs[..],
        );
        output_layer.update_cost_gradients_simd(learn_data_output);

        // Update for hidden layers
        for i in (0..(self.layers.len() - 1)).rev() {
            let (left, right) = self.layer_learn_data.split_at_mut(i + 1);
            let learn_data_hidden = &mut left[i];
            let learn_data_hidden_next = &right[0];

            let hidden_layer: &Layer = &self.layers[i];
            let prev_layer = &self.layers[i + 1];
            hidden_layer.calculate_hidden_layer_node_cost_values(
                learn_data_hidden,
                prev_layer,
                &learn_data_hidden_next.node_values,
            );

            let mut_hidden_layer = &mut self.layers[i];
            mut_hidden_layer.update_cost_gradients_simd(learn_data_hidden);
        }

        output_inputs
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

    pub fn calculate_outputs(&self, inputs: &[f32]) -> Vec<f32> {
        let mut current_inputs = inputs.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            current_inputs = layer.calculate_outputs_simd(&current_inputs);
        }

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

    fn calculate_cost_datapoint(&self, datapoint: &DataPoint) -> f32 {
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

    pub fn calculate_cost(&self, data: &[DataPoint]) -> f32 {
        if data.len() <= 0 {
            panic!("Input data was len: {}", data.len());
        }

        let mut cost: f32 = 0.0;
        for datapoint in &data[..] {
            cost += self.calculate_cost_datapoint(datapoint);
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
        log::info!("----------NEURAL NETWORK----------\n");
        log::info!("{}", self.to_string());
        log::info!("----------------------------------\n");
    }
}
