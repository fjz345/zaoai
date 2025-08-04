use crate::layer::*;
use crate::zneural_network::is_correct::IsCorrectFn;
use crate::zneural_network::thread::TrainingThreadPayload;
use crate::zneural_network::training::{AIResultMetadata, DatasetUsage, FloatDecay, TestResults};

use super::datapoint::DataPoint;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::Duration;
use wide::f32x8;

impl LayerLearnData {
    fn new(layer: &Layer) -> LayerLearnData {
        LayerLearnData {
            inputs: vec![0.0; layer.num_in_nodes],
            weighted_inputs: vec![0.0; layer.num_out_nodes],
            activation_values: vec![0.0; layer.num_out_nodes],
            node_values: vec![0.0; layer.num_out_nodes],
            dropout_mask: None,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, bincode::Encode, bincode::Decode)]
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct NeuralNetwork {
    pub graph_structure: GraphStructure,
    pub layers: Vec<Layer>,
    pub last_test_results: Option<TestResults>,
    pub is_softmax_output: bool,
    layer_learn_data: Vec<LayerLearnData>,
    version: u8,
    layer_activation_function: ActivationFunctionType,
}

pub type NNGreatestValueResult = (usize, f32);
pub type NNOutputs = Vec<f32>;
pub type NNOutputsRef = [f32];
pub type NNExpectedOutputs = Vec<f32>;
pub type NNExpectedOutputsRef = [f32];

impl NeuralNetwork {
    const VERSION: u8 = 2;
    pub fn new(
        graph_structure: GraphStructure,
        layer_activation: ActivationFunctionType,
    ) -> NeuralNetwork {
        let mut layers: Vec<Layer> = Vec::new();
        let mut prev_out_size = graph_structure.input_nodes;

        // Input nodes are not layers in the neural network.

        // Create Hidden layers
        for i in &graph_structure.hidden_layers[..] {
            layers.push(Layer::new(prev_out_size, *i, layer_activation, Some(0.5)));
            prev_out_size = *i;
        }

        // Create Output layer
        layers.push(Layer::new(
            prev_out_size,
            graph_structure.output_nodes,
            layer_activation,
            None,
        ));

        let mut layer_learn_data: Vec<LayerLearnData> = Vec::new();
        for i in 0..layers.len() {
            let layer: &Layer = &layers[i];
            layer_learn_data.push(LayerLearnData::new(&layer));
        }

        NeuralNetwork {
            graph_structure,
            layers,
            last_test_results: None,
            layer_learn_data,
            version: Self::VERSION,
            is_softmax_output: false,
            layer_activation_function: layer_activation,
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    fn cross_entropy_loss(predicted: &[f32], expected: &[f32]) -> f32 {
        Self::cross_entropy_loss_multiclass(predicted, expected)
    }

    fn cross_entropy_loss_multiclass(predicted: &[f32], expected: &[f32]) -> f32 {
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

    fn cross_entropy_loss_binary(predicted: &[f32], expected: &[f32]) -> f32 {
        let epsilon = 1e-12;

        predicted
            .iter()
            .zip(expected.iter())
            .map(|(p, e)| {
                let p_clamped = p.max(epsilon).min(1.0 - epsilon);
                -(e * p_clamped.ln() + (1.0 - e) * (1.0 - p_clamped).ln())
            })
            .sum()
    }

    fn apply_dropout(inputs: &mut [f32], mask: &mut Vec<f32>, dropout_prob: f32) {
        let keep_prob = 1.0 - dropout_prob;
        let mut rng = rand::thread_rng();

        for (i, input) in inputs.iter_mut().enumerate() {
            if rng.gen::<f32>() < dropout_prob {
                mask[i] = 0.0;
                *input = 0.0;
            } else {
                mask[i] = 1.0 / keep_prob; // scale up remaining activations
                *input *= mask[i];
            }
        }
    }

    pub fn learn_batch(
        &mut self,
        batch_data: &[DataPoint],
        learn_rate: f32,
        batch_data_cost: &mut f32,
        batch_data_loss: &mut f32,
    ) -> Vec<Vec<f32>> {
        // let new_metadata = AIResultMetadata::new(DatasetUsage::Training);

        if (batch_data.len() <= 0) {
            panic!("DataPoints length was 0");
        }

        let mut total_cost = 0.0;
        let mut total_loss = 0.0;
        let mut batch_data_outputs = Vec::new();
        for datapoint in batch_data {
            // Todo: make functions forward/backward for simplicity.
            let datapoint_outputs = self.learn_calculate_outputs(datapoint);
            let loss = Self::cross_entropy_loss(&datapoint_outputs, &datapoint.expected_outputs);
            let mut cost = 0.0;
            for (i, datapoint_output) in datapoint_outputs.iter().enumerate() {
                cost += node_cost(*datapoint_output, datapoint.expected_outputs[i]);
            }

            total_loss += loss;
            total_cost += cost;

            batch_data_outputs.push(datapoint_outputs);
        }
        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (batch_data.len() as f32));
        self.clear_all_cost_gradients();

        *batch_data_cost = total_cost;
        *batch_data_loss = total_loss;
        log::trace!("Cost: {}", batch_data_cost);

        batch_data_outputs
    }

    pub fn learn_epoch(
        &mut self,
        epoch_index: usize,
        training_data: &[DataPoint],
        batch_size: usize,
        learn_rate: f32,
        is_correct_fn: IsCorrectFn,
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

        let mut cur_index = 0;
        let len = training_data.len();

        let mut process_batch =
            |data: &[DataPoint], batch_num: usize, total_batches: usize, cur_index: usize| {
                log::trace!(
                    "Training... @{} #[{}/{}] (#{} - #{})",
                    epoch_index + 1,
                    batch_num + 1,
                    total_batches,
                    cur_index,
                    cur_index + data.len(),
                );

                let mut batch_data_cost = 0.0;
                let mut batch_data_loss = 0.0;
                let batch_data_outputs =
                    self.learn_batch(data, learn_rate, &mut batch_data_cost, &mut batch_data_loss);

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
                        is_correct_fn,
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
        is_correct_fn: IsCorrectFn,
        new_metadata: &mut AIResultMetadata,
    ) {
        for (i, data) in epoch_data.iter().enumerate() {
            let datapoint_output = &epoch_data_outputs[i];

            let correct = is_correct_fn.call(datapoint_output, &data.expected_outputs);

            if correct {
                new_metadata.true_positives += 1;
                new_metadata.positive_instances += 1;
            } else {
                new_metadata.false_positives += 1;
                new_metadata.negative_instances += 1;
            }
        }

        new_metadata.cost = epoch_data_cost as f64;
    }

    pub fn learn<T: Fn() -> bool>(
        &mut self,
        training_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
        learn_rate_decay: Option<FloatDecay>,
        learn_rate_decay_rate: f32,
        tx_training_metadata: Option<&Sender<TrainingThreadPayload>>,
        is_correct_fn: IsCorrectFn,
        eval_abort_fn: Option<T>,
    ) {
        assert!(learn_rate > 0.0);
        assert!(training_data.len() > 0);
        assert!(batch_size > 0);

        for e in 0..num_epochs {
            log::trace!(
                "Training...Learn Epoch Started [@{}/@{}]",
                e + 1,
                num_epochs
            );
            let mut metadata: AIResultMetadata =
                AIResultMetadata::new(DatasetUsage::Training, 0.0, 0.0);

            let maybe_decayed_learn_rate = learn_rate_decay
                .as_ref()
                .and_then(|f| Some(f.decay(learn_rate, e)))
                .unwrap_or(learn_rate);
            self.learn_epoch(
                e,
                &training_data,
                batch_size,
                maybe_decayed_learn_rate,
                is_correct_fn,
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

            if let Some(post_fn) = &eval_abort_fn {
                let abort_recv = post_fn();
                if abort_recv {
                    log::info!("Training thread received abort signal.");
                    break;
                }
            }
        }

        log::info!("Training...Complete! [@{} Epochs]", num_epochs);
    }

    pub fn learn_calculate_outputs(&mut self, datapoint: &DataPoint) -> Vec<f32> {
        // Forward pass
        let mut current_inputs = datapoint.inputs.to_vec();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            #[cfg(feature = "simd")]
            {
                current_inputs = layer.calculate_outputs_learn_simd(
                    &mut self.layer_learn_data[i],
                    &mut current_inputs,
                );
            }
            #[cfg(not(feature = "simd"))]
            {
                current_inputs = layer
                    .calculate_outputs_learn(&mut self.layer_learn_data[i], &mut current_inputs);
            }

            // Dropout
            if let Some(prob) = layer.dropout_prob {
                let learn_data = &mut self.layer_learn_data[i];
                if learn_data.dropout_mask.is_none() {
                    learn_data.dropout_mask = Some(vec![0.0; current_inputs.len()]);
                }
                Self::apply_dropout(
                    &mut current_inputs,
                    learn_data.dropout_mask.as_mut().unwrap(),
                    prob,
                );
            }
        }
        let output_inputs = current_inputs;

        // Backward pass (backpropagation)
        self.backpropagation(datapoint);

        output_inputs
    }

    fn backpropagation(&mut self, datapoint: &DataPoint) {
        let layer_len = self.layers.len();

        // Output layer error & gradients
        let output_layer = &mut self.layers[layer_len - 1];
        let learn_data_output = &mut self.layer_learn_data[layer_len - 1];
        output_layer.calculate_output_layer_node_cost_values(
            learn_data_output,
            &datapoint.expected_outputs,
        );
        #[cfg(feature = "simd")]
        {
            output_layer.update_cost_gradients_simd(learn_data_output);
        }
        #[cfg(not(feature = "simd"))]
        {
            output_layer.update_cost_gradients(learn_data_output);
        }

        // Hidden layers error & gradients, back to front
        for i in (0..layer_len - 1).rev() {
            let (left, right) = self.layer_learn_data.split_at_mut(i + 1);
            let learn_data_hidden = &mut left[i];
            let learn_data_hidden_next = &right[0];

            let hidden_layer = &self.layers[i];
            let next_layer = &self.layers[i + 1];
            hidden_layer.calculate_hidden_layer_node_cost_values(
                learn_data_hidden,
                next_layer,
                &learn_data_hidden_next.node_values,
            );

            let mut_hidden_layer = &mut self.layers[i];

            #[cfg(feature = "simd")]
            {
                mut_hidden_layer.update_cost_gradients_simd(learn_data_hidden);
            }
            #[cfg(not(feature = "simd"))]
            {
                mut_hidden_layer.update_cost_gradients(learn_data_hidden);
            }
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

    pub fn calculate_outputs(&self, inputs: &[f32]) -> Vec<f32> {
        let mut current_inputs = inputs.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            #[cfg(feature = "simd")]
            {
                current_inputs = layer.calculate_outputs_simd(&current_inputs);
            }
            #[cfg(not(feature = "simd"))]
            {
                current_inputs = layer.calculate_outputs(&current_inputs);
            }
        }

        if self.is_softmax_output {
            current_inputs = ActivationFunctionType::apply_softmax(&current_inputs)
        }
        current_inputs
    }

    // returns index of max value, max value
    pub fn determine_output_greatest_value_result(inputs: &[f32]) -> NNGreatestValueResult {
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

    pub fn calculate_costs(&self, data: &[DataPoint]) -> f32 {
        if data.len() <= 0 {
            panic!("Input data was len: {}", data.len());
        }

        #[cfg(feature = "simd")]
        {
            self.calculate_cost_simd(data)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.calculate_cost(data)
        }
    }

    fn calculate_cost(&self, data: &[DataPoint]) -> f32 {
        let mut cost: f32 = 0.0;
        for datapoint in &data[..] {
            cost += self.calculate_cost_datapoint(datapoint);
        }

        cost / (data.len() as f32)
    }

    #[cfg(feature = "simd")]
    fn calculate_cost_simd(&self, data: &[DataPoint]) -> f32 {
        let mut total_cost = 0.0;
        let output_layer = self.layers.last().unwrap();
        let num_outputs = output_layer.num_out_nodes;

        for datapoint in data {
            let output = self.calculate_outputs(&datapoint.inputs);

            let mut i = 0;
            let mut sum = f32x8::splat(0.0);
            while i + 8 <= num_outputs {
                let pred = f32x8::from(&output[i..i + 8]);
                let expected = f32x8::from(&datapoint.expected_outputs[i..i + 8]);
                let delta = pred - expected;
                sum += delta * delta;
                i += 8;
            }

            // Horizontal sum for SIMD part
            let mut cost = sum.reduce_add();

            // Tail values (non-multiple of 8)
            while i < num_outputs {
                cost += node_cost(output[i], datapoint.expected_outputs[i]);
                i += 1;
            }

            total_cost += cost;
        }

        total_cost / (data.len() as f32)
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
        let last_test_result_string = if let Some(last_test_results) = &self.last_test_results {
            format!("{}", last_test_results)
        } else {
            "".to_string()
        };
        let print_string: String = format!(
            "\
        Graph Structure: {}\n\
        Last Test Results: {}\n",
            self.graph_structure.to_string(),
            last_test_result_string
        );

        print_string
    }

    pub fn print(&self) {
        log::info!("----------NEURAL NETWORK----------\n");
        log::info!("{}", self.to_string());
        log::info!("----------------------------------\n");
    }
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();
pub fn save_neural_network<P: AsRef<Path>>(nn: &NeuralNetwork, path: P) -> std::io::Result<()> {
    // Create parent directories if they don't exist
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let encoded: Vec<u8> = bincode::encode_to_vec(&nn, BINCODE_CONFIG).unwrap();
    let mut file = File::create(path)?;
    file.write(&encoded)?;
    Ok(())
}

pub fn load_neural_network(path: &str) -> std::io::Result<NeuralNetwork> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (decoded, len): (NeuralNetwork, usize) =
        bincode::decode_from_slice(&buffer[..], BINCODE_CONFIG)
            .expect("load_neural_network failed, decoding failed.");

    assert_eq!(len, buffer.len()); // read all bytes
    Ok(decoded)
}
