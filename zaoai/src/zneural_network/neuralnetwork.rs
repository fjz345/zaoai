use crate::layer::*;
use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::{cross_entropy_loss_multiclass, mse, CostFunction};
use crate::zneural_network::datapoint::TrainingData;
use crate::zneural_network::is_correct::ConfusionEvaluator;
use crate::zneural_network::thread::TrainingThreadPayload;
use crate::zneural_network::training::{
    test_nn, AIResultMetadata, DatasetUsage, FloatDecay, TestResults,
};

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
    cost_fn: CostFunction,
}

pub type NNOutputs = Vec<f32>;
impl NeuralNetwork {
    const VERSION: u8 = 2;
    pub fn new(
        graph_structure: GraphStructure,
        layer_activation: ActivationFunctionType,
        cost_fn: CostFunction,
        dropout_prob: Option<f32>,
        weight_init: WeightInit,
        bias_init: BiasInit,
    ) -> NeuralNetwork {
        let mut layers: Vec<Layer> = Vec::new();
        let mut prev_out_size = graph_structure.input_nodes;

        // Input nodes are not layers in the neural network.

        // Create Hidden layers
        for i in &graph_structure.hidden_layers[..] {
            layers.push(Layer::new(
                prev_out_size,
                *i,
                layer_activation,
                dropout_prob,
                weight_init,
                bias_init,
            ));
            prev_out_size = *i;
        }

        // Create Output layer
        layers.push(Layer::new(
            prev_out_size,
            graph_structure.output_nodes,
            layer_activation,
            None,
            weight_init,
            bias_init,
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
            cost_fn: cost_fn,
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
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
        if (batch_data.len() <= 0) {
            panic!("DataPoints length was 0");
        }

        let mut total_cost = 0.0;
        let mut last_loss = 0.0; // last batches cost
        let mut batch_data_outputs = Vec::with_capacity(batch_data.len());
        for (i, datapoint) in batch_data.iter().enumerate() {
            let datapoint_outputs = self.learn_calculate_outputs(datapoint);
            let loss =
                cross_entropy_loss_multiclass(&datapoint_outputs, &datapoint.expected_outputs);
            let cost = self.cost_function(&datapoint_outputs, &datapoint.expected_outputs);

            total_cost += cost;
            if i == batch_data.len() - 1 {
                last_loss = cost;
            }

            batch_data_outputs.push(datapoint_outputs);
        }
        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (batch_data.len() as f32));
        self.clear_all_cost_gradients();

        *batch_data_cost = total_cost / batch_data.len() as f32;
        *batch_data_loss = last_loss;
        log::trace!("Cost: {}", batch_data_cost);
        log::trace!("Last Loss: {}", batch_data_loss);

        batch_data_outputs
    }

    pub fn learn_epoch(
        &mut self,
        epoch_index: usize,
        training_data: &[DataPoint],
        batch_size: usize,
        learn_rate: f32,
        is_correct_fn: ConfusionEvaluator,
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
                        learn_rate,
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
        is_correct_fn: ConfusionEvaluator,
        new_metadata: &mut AIResultMetadata,
    ) {
        for (i, data) in epoch_data.iter().enumerate() {
            let datapoint_output = &epoch_data_outputs[i];

            let confusion_cat = is_correct_fn.evaluate(datapoint_output, &data.expected_outputs);
            match confusion_cat {
                super::is_correct::ConfusionCategory::TruePositive => {
                    new_metadata.true_positives += 1;
                }
                super::is_correct::ConfusionCategory::TrueNegative => {
                    new_metadata.true_negatives += 1
                }
                super::is_correct::ConfusionCategory::FalsePositive => {
                    new_metadata.false_positives += 1
                }
                super::is_correct::ConfusionCategory::FalseNegative => {
                    new_metadata.false_negatives += 1
                }
            }
        }
        new_metadata.cost = epoch_data_cost as f64;
    }

    pub fn learn<T: Fn() -> bool>(
        &mut self,
        training_data: &[DataPoint],
        validation_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
        learn_rate_decay: Option<FloatDecay>,
        learn_rate_decay_rate: f32,
        tx_training_metadata: Option<&Sender<TrainingThreadPayload>>,
        tx_validation_metadata: Option<&Sender<TrainingThreadPayload>>,
        is_correct_fn: ConfusionEvaluator,
        eval_abort_fn: Option<T>,
        validation_each_epoch: usize,
    ) {
        assert!(learn_rate > 0.0);
        assert!(training_data.len() > 0);
        assert!(batch_size > 0);

        for e in 0..num_epochs {
            let mut test_and_send_payload =
                |tx: &Sender<TrainingThreadPayload>, data: &[DataPoint], payload_index: usize| {
                    // Send training meta data before training for baseline graph point
                    if let Some(test_results) = test_nn(self, data, is_correct_fn, None, None).ok()
                    {
                        let mut initial_metadata = AIResultMetadata::from_accuracy(
                            test_results.accuracy.unwrap_or_default() as f64,
                            test_results.results.len(),
                        );
                        initial_metadata.cost = test_results.cost as f64;
                        initial_metadata.last_loss = test_results.cost as f64;
                        initial_metadata.learn_rate = learn_rate;

                        let payload = TrainingThreadPayload {
                            payload_index: payload_index,
                            payload_max_index: num_epochs - 1,
                            training_metadata: initial_metadata,
                        };
                        tx.send(payload);
                    }
                };
            if e == 0 {
                if let Some(tx_testing_metadata) = tx_training_metadata {
                    test_and_send_payload(tx_testing_metadata, training_data, e);
                }
            }
            if validation_each_epoch != 0 && e % validation_each_epoch == 0 {
                if let Some(tx_validation_metadata) = tx_validation_metadata {
                    log::trace!("Testing and sending validation data...");
                    test_and_send_payload(tx_validation_metadata, validation_data, e);
                }
            }

            log::trace!(
                "Training...Learn Epoch Started [@{}/@{}]",
                e + 1,
                num_epochs
            );
            let mut metadata: AIResultMetadata =
                AIResultMetadata::new(DatasetUsage::Training, 0.0, 0.0, 0.0);

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
                    payload_index: e + 1,
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
        let outputs = self.forward(&datapoint.inputs);
        self.backpropagation(datapoint);
        outputs
    }

    fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut current_inputs = inputs.to_vec();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let learn_data = &mut self.layer_learn_data[i];
            learn_data.inputs.reserve(current_inputs.len());

            current_inputs = if cfg!(feature = "simd") {
                layer.calculate_outputs_learn_simd(&mut current_inputs, learn_data)
            } else {
                layer.calculate_outputs_learn(&mut current_inputs, learn_data)
            };

            if let Some(prob) = layer.dropout_prob {
                let mask = learn_data
                    .dropout_mask
                    .get_or_insert_with(|| vec![0.0; current_inputs.len()]);
                Self::apply_dropout(&mut current_inputs, mask, prob);
            }
        }

        current_inputs
    }

    fn backpropagation(&mut self, datapoint: &DataPoint) {
        let last = self.layers.len() - 1;

        // --- Output layer ---
        {
            let (layer, learn_data) = self.layers.split_at_mut(last);
            let output_layer = &mut learn_data[0];
            let learn_data_output = &mut self.layer_learn_data[last];

            output_layer.calculate_output_layer_node_cost_values(
                learn_data_output,
                &datapoint.expected_outputs,
                self.cost_fn,
            );
            Self::update_gradients(output_layer, learn_data_output);
        }

        // --- Hidden layers (reverse) ---
        for i in (0..last).rev() {
            let (left, right) = self.layer_learn_data.split_at_mut(i + 1);
            let learn_data_hidden = &mut left[i];
            let learn_data_next = &right[0];

            let hidden_layer = &self.layers[i];
            let next_layer = &self.layers[i + 1];
            hidden_layer.calculate_hidden_layer_node_cost_values(
                learn_data_hidden,
                next_layer,
                &learn_data_next.node_values,
            );

            let mut_layer = &mut self.layers[i];
            Self::update_gradients(mut_layer, learn_data_hidden);
        }
    }

    #[inline]
    fn update_gradients(layer: &mut Layer, learn_data: &mut LayerLearnData) {
        #[cfg(feature = "simd")]
        layer.update_cost_gradients_simd(learn_data);

        #[cfg(not(feature = "simd"))]
        layer.update_cost_gradients(learn_data);
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

    fn cost_function(&self, predicted: &[f32], expected: &[f32]) -> f32 {
        self.cost_fn.call(predicted, expected)
    }

    fn calculate_cost_datapoint(&self, datapoint: &DataPoint) -> f32 {
        // Prediction cost
        let outputs = self.calculate_outputs(&datapoint.inputs);
        let cost = self.cost_function(&outputs, &datapoint.expected_outputs);

        // L2 regularization penalty (sum of squared weights)
        let l2_penalty: f32 = self
            .layers
            .iter()
            .flat_map(|layer| layer.weights.iter())
            .flat_map(|matrix| matrix.iter())
            .map(|w| w.powi(2))
            .sum();

        const LAMBDA: f32 = 0.001;
        cost + LAMBDA * l2_penalty
    }

    pub fn calculate_costs(&self, data: &[DataPoint]) -> f32 {
        assert!(!data.is_empty(), "Input data was empty");

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
        let total: f32 = data
            .iter()
            .map(|dp| self.calculate_cost_datapoint(dp))
            .sum();

        total / (data.len() as f32)
    }

    #[cfg(feature = "simd")]
    fn calculate_cost_simd(&self, data: &[DataPoint]) -> f32 {
        let output_layer = self.layers.last().unwrap();
        let num_outputs = output_layer.num_out_nodes;

        let total_cost: f32 = data
            .iter()
            .map(|datapoint| {
                let output = self.calculate_outputs(&datapoint.inputs);
                let mut sum = f32x8::splat(0.0);
                let mut i = 0;

                while i + 8 <= num_outputs {
                    let pred = f32x8::from(&output[i..i + 8]);
                    let expected = f32x8::from(&datapoint.expected_outputs[i..i + 8]);
                    sum += self.cost_fn.call_simd(pred, expected);
                    i += 8;
                }

                let mut cost = sum.reduce_add();
                if i < num_outputs {
                    cost += self
                        .cost_fn
                        .call(&output[i..], &datapoint.expected_outputs[i..]);
                }
                cost
            })
            .sum();

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
