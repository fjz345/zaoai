use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::CostFunction;
use crate::zneural_network::cpu::layer::{calculate_cost, Layer, LayerLearnData};
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::is_correct::ConfusionEvaluator;
use crate::zneural_network::thread::TrainingThreadPayload;
use crate::zneural_network::training::{AIResultMetadata, DatasetUsage, FloatDecay, TestResults};

use crate::weight_bias::{BiasInit, WeightInit};
use rand::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use zaoai_types::ai_labels::LayerTypeCPU;

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
    pub hidden_layers: Vec<usize>,
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
            if i >= 1 {
                result_string += ", ";
            }
            result_string += layer.to_string().as_str();
        }
        result_string
    }
}

pub struct NeuralNetworkPingPong {
    pub current: Vec<LayerTypeCPU>, // in
    pub next: Vec<LayerTypeCPU>,    // out
}

impl NeuralNetworkPingPong {
    pub fn new(max_layer_nodes: usize) -> Self {
        Self {
            current: Vec::with_capacity(max_layer_nodes),
            next: Vec::with_capacity(max_layer_nodes),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct NeuralNetworkCPU {
    pub graph_structure: GraphStructure,
    pub layers: Vec<Layer>,
    pub last_test_results: Option<TestResults>,
    pub is_softmax_output: bool,
    layer_learn_data: Vec<LayerLearnData>,
    version: u8,
    layer_activation_function: ActivationFunctionType,
    pub cost_fn: CostFunction,
}

pub type NNOutputs<T> = Vec<T>;
impl NeuralNetworkCPU {
    const VERSION: u8 = 2;
    pub fn new(
        graph_structure: GraphStructure,
        layer_activation: ActivationFunctionType,
        cost_fn: CostFunction,
        dropout_prob: Option<LayerTypeCPU>,
        weight_init: WeightInit,
        bias_init: BiasInit,
        is_softmax_output: bool,
    ) -> Self {
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

        let output_activation = if is_softmax_output {
            ActivationFunctionType::Linear
        } else {
            layer_activation
        };

        // Create Output layer
        layers.push(Layer::new(
            prev_out_size,
            graph_structure.output_nodes,
            output_activation,
            None,
            weight_init,
            bias_init,
        ));

        let mut layer_learn_data: Vec<LayerLearnData> = Vec::new();
        for i in 0..layers.len() {
            let layer: &Layer = &layers[i];
            layer_learn_data.push(LayerLearnData::new(&layer));
        }

        NeuralNetworkCPU {
            graph_structure,
            layers,
            last_test_results: None,
            layer_learn_data,
            version: Self::VERSION,
            is_softmax_output: is_softmax_output,
            layer_activation_function: layer_activation,
            cost_fn: cost_fn,
        }
    }

    pub fn max_layer_nodes(&self) -> usize {
        let hidden_layers_max = *self
            .graph_structure
            .hidden_layers
            .iter()
            .max()
            .unwrap_or(&0);
        let max = self
            .graph_structure
            .input_nodes
            .max(self.graph_structure.output_nodes)
            .max(hidden_layers_max);
        max
    }

    pub fn get_parameters_num(&self) -> usize {
        let mut total_params = 0;
        let mut prev_nodes = self.graph_structure.input_nodes;

        for &nodes in &self.graph_structure.hidden_layers {
            total_params += (prev_nodes * nodes) + nodes;
            prev_nodes = nodes;
        }

        let out_nodes = self.graph_structure.output_nodes;
        total_params += (prev_nodes * out_nodes) + out_nodes;

        total_params
    }

    pub fn get_parameters_unit_size(&self) -> usize {
        std::mem::size_of::<LayerTypeCPU>()
    }

    fn apply_dropout(
        inputs: &mut [LayerTypeCPU],
        mask: &mut Vec<LayerTypeCPU>,
        dropout_prob: LayerTypeCPU,
    ) {
        let keep_prob = 1.0 - dropout_prob;
        let mut rng = rand::thread_rng();

        for (i, input) in inputs.iter_mut().enumerate() {
            if rng.gen::<LayerTypeCPU>() < dropout_prob {
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
        learn_rate: LayerTypeCPU,
        batch_data_cost: &mut LayerTypeCPU,
        batch_data_loss: &mut LayerTypeCPU,
        pingpong: &mut NeuralNetworkPingPong,
    ) -> Vec<Vec<LayerTypeCPU>> {
        if batch_data.len() <= 0 {
            panic!("DataPoints length was 0");
        }

        let mut total_cost = 0.0 as LayerTypeCPU;
        let mut last_loss = 0.0 as LayerTypeCPU; // last batches cost
        let mut batch_data_outputs = Vec::with_capacity(batch_data.len());
        for (i, datapoint) in batch_data.iter().enumerate() {
            self.learn_datapoint(datapoint, pingpong);
            // let loss =
            //     cross_entropy_loss_multiclass(&datapoint_outputs, &datapoint.expected_outputs);
            let cost = self.cost_function(&pingpong.next, &datapoint.expected_outputs);

            total_cost += cost;
            if i == batch_data.len() - 1 {
                last_loss = cost;
            }
            batch_data_outputs.push(pingpong.next.clone());
        }
        // Adjust weights & biases
        self.apply_all_cost_gradients(learn_rate / (batch_data.len() as LayerTypeCPU));
        self.clear_all_cost_gradients();

        *batch_data_cost = total_cost / batch_data.len() as LayerTypeCPU;
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
        learn_rate: LayerTypeCPU,
        is_correct_fn: ConfusionEvaluator,
        mut epoch_metadata: Option<&mut AIResultMetadata>,
        pingpong: &mut NeuralNetworkPingPong,
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
                let batch_data_outputs = self.learn_batch(
                    data,
                    learn_rate,
                    &mut batch_data_cost,
                    &mut batch_data_loss,
                    pingpong,
                );

                if let Some(metadata) = epoch_metadata.as_mut() {
                    let mut new_metadata = AIResultMetadata::new(
                        DatasetUsage::Training,
                        batch_data_cost as LayerTypeCPU,
                        batch_data_loss as LayerTypeCPU,
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
        epoch_data_outputs: &Vec<Vec<LayerTypeCPU>>,
        epoch_data_cost: LayerTypeCPU,
        is_correct_fn: ConfusionEvaluator,
        new_metadata: &mut AIResultMetadata,
    ) {
        for (i, data) in epoch_data.iter().enumerate() {
            let datapoint_output = &epoch_data_outputs[i];

            let confusion_cat = is_correct_fn.evaluate(datapoint_output, &data.expected_outputs);
            match confusion_cat {
                crate::is_correct::ConfusionCategory::TruePositive => {
                    new_metadata.true_positives += 1;
                }
                crate::is_correct::ConfusionCategory::TrueNegative => {
                    new_metadata.true_negatives += 1
                }
                crate::is_correct::ConfusionCategory::FalsePositive => {
                    new_metadata.false_positives += 1
                }
                crate::is_correct::ConfusionCategory::FalseNegative => {
                    new_metadata.false_negatives += 1
                }
            }
        }
        new_metadata.cost = epoch_data_cost as LayerTypeCPU;
    }

    pub fn learn<T: Fn() -> bool>(
        &mut self,
        training_data: &[DataPoint],
        validation_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: LayerTypeCPU,
        learn_rate_decay: Option<FloatDecay>,
        _learn_rate_decay_rate: LayerTypeCPU,
        tx_training_metadata: Option<&Sender<TrainingThreadPayload>>,
        tx_validation_metadata: Option<&Sender<TrainingThreadPayload>>,
        is_correct_fn: ConfusionEvaluator,
        eval_abort_fn: Option<T>,
        validation_each_epoch: usize,
    ) {
        assert!(learn_rate > 0.0);
        assert!(training_data.len() > 0);
        assert!(batch_size > 0);

        let pingpong = &mut NeuralNetworkPingPong::new(self.max_layer_nodes());

        for e in 0..num_epochs {
            let mut test_nn_and_send_payload =
                |tx: &Sender<TrainingThreadPayload>, data: &[DataPoint], payload_index: usize| {
                    if let Some(test_results) =
                        test_nn_cpu(self, data, is_correct_fn, None, None, pingpong).ok()
                    {
                        let mut result_metadata = AIResultMetadata::from_accuracy(
                            test_results.accuracy.unwrap_or_default() as f64,
                            test_results.results.len(),
                        );
                        result_metadata.cost = test_results.cost as LayerTypeCPU;
                        result_metadata.last_loss = test_results.cost as LayerTypeCPU;
                        result_metadata.learn_rate = learn_rate;

                        let payload = TrainingThreadPayload {
                            payload_index: payload_index,
                            payload_max_index: num_epochs - 1,
                            training_metadata: result_metadata,
                        };
                        if let Err(e) = tx.send(payload) {
                            log::error!("Failed to send training metadata through channel: {}", e);
                        }
                    }
                };
            if e == 0 {
                if let Some(tx_testing_metadata) = tx_training_metadata {
                    // Send training meta data before training for baseline graph point
                    test_nn_and_send_payload(tx_testing_metadata, training_data, e);
                }
            }
            if validation_each_epoch != 0 && e % validation_each_epoch == 0 {
                if let Some(tx_validation_metadata) = tx_validation_metadata {
                    log::trace!("Testing and sending validation data...");
                    test_nn_and_send_payload(tx_validation_metadata, validation_data, e);
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
                pingpong,
            );

            if tx_training_metadata.is_some() {
                let payload = TrainingThreadPayload {
                    payload_index: e + 1,
                    payload_max_index: num_epochs - 1,
                    training_metadata: metadata,
                };
                if let Err(e) = tx_training_metadata.unwrap().send(payload) {
                    log::error!("Failed to send training metadata through channel: {}", e)
                };
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

    pub fn learn_datapoint(&mut self, datapoint: &DataPoint, pingpong: &mut NeuralNetworkPingPong) {
        self.forward_learn(&datapoint.inputs, pingpong);
        self.backpropagation_learn(datapoint);
        // Now results should be in pingpong.next
    }

    fn cost_function(&self, predicted: &[LayerTypeCPU], expected: &[LayerTypeCPU]) -> LayerTypeCPU {
        self.cost_fn.call(predicted, expected)
    }

    fn forward_learn(&mut self, inputs: &[LayerTypeCPU], pingpong: &mut NeuralNetworkPingPong) {
        pingpong.current.clear();
        pingpong.current.extend_from_slice(inputs);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            pingpong.next.resize(layer.num_out_nodes, 0.0);

            let learn_data = &mut self.layer_learn_data[i];
            layer.calculate_outputs_learn(&pingpong.current, &mut pingpong.next, learn_data);

            if let Some(prob) = layer.dropout_prob {
                let mask = learn_data
                    .dropout_mask
                    .get_or_insert_with(|| vec![0.0; pingpong.next.len()]);
                Self::apply_dropout(&mut pingpong.next, mask, prob);
            }
            std::mem::swap(&mut pingpong.current, &mut pingpong.next);
        }
        if self.is_softmax_output {
            let max_value = pingpong
                .current
                .iter()
                .copied()
                .fold(LayerTypeCPU::NEG_INFINITY, LayerTypeCPU::max);

            let mut sum = 0.0;
            for value in pingpong.current.iter_mut() {
                *value = (*value - max_value).exp();
                sum += *value;
            }
            if sum > 0.0 {
                for value in pingpong.current.iter_mut() {
                    *value /= sum;
                }
            }
        }
        // For simplicity, keep input at current, output at next
        std::mem::swap(&mut pingpong.current, &mut pingpong.next);
    }

    fn backpropagation_learn(&mut self, datapoint: &DataPoint) {
        let last = self.layers.len() - 1;

        // --- Output layer ---
        {
            let (_layer, learn_data) = self.layers.split_at_mut(last);
            let output_layer = &mut learn_data[0];
            let learn_data_output = &mut self.layer_learn_data[last];

            #[cfg(feature = "simd")]
            output_layer.calculate_output_layer_node_cost_values_simd(
                learn_data_output,
                &datapoint.expected_outputs,
                self.cost_fn,
            );
            #[cfg(not(feature = "simd"))]
            output_layer.calculate_output_layer_node_cost_values(
                learn_data_output,
                &datapoint.expected_outputs,
                self.cost_fn,
            );
            output_layer.update_cost_gradients(learn_data_output);
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
            mut_layer.update_cost_gradients(learn_data_hidden);
        }
    }

    fn apply_all_cost_gradients(&mut self, learn_rate: LayerTypeCPU) {
        for layer in self.layers.iter_mut() {
            layer.apply_cost_gradient(learn_rate);
        }
    }
    fn clear_all_cost_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cost_gradient();
        }
    }

    // pub fn validate(&self) -> bool {
    //     let mut is_valid: bool = true;

    //     // Validate Graph Structure
    //     if !self.graph_structure.validate() {
    //         is_valid = false;
    //     }

    //     // Ensure that the layers input/output numbers match
    //     let mut prev_out_size: usize = self.graph_structure.input_nodes;
    //     for layer in &self.layers[..] {
    //         if layer.num_in_nodes != prev_out_size {
    //             is_valid = false;
    //             break;
    //         }

    //         prev_out_size = layer.num_out_nodes;
    //     }

    //     // TODO: validate In_nodes & out_nodes with graph_strucutre values also

    //     is_valid
    // }

    fn format_count(n: usize) -> String {
        let n = n as f64;
        if n >= 1e12 {
            format!("{:.1}T", n / 1e12).replace(".0", "")
        } else if n >= 1e9 {
            format!("{:.1}B", n / 1e9).replace(".0", "")
        } else if n >= 1e6 {
            format!("{:.1}M", n / 1e6).replace(".0", "")
        } else if n >= 1e3 {
            format!("{:.1}K", n / 1e3).replace(".0", "")
        } else {
            n.to_string()
        }
    }

    fn format_bytes(bytes: usize) -> String {
        const KIB: f64 = 1024.0;
        const MIB: f64 = KIB * 1024.0;
        const GIB: f64 = MIB * 1024.0;
        const TIB: f64 = GIB * 1024.0;

        let bytes = bytes as f64;

        if bytes >= TIB {
            format!("{:.1} TiB", bytes / TIB).replace(".0", "")
        } else if bytes >= GIB {
            format!("{:.1} GiB", bytes / GIB).replace(".0", "")
        } else if bytes >= MIB {
            format!("{:.1} MiB", bytes / MIB).replace(".0", "")
        } else if bytes >= KIB {
            format!("{:.1} KiB", bytes / KIB).replace(".0", "")
        } else {
            format!("{} B", bytes as usize)
        }
    }

    pub fn to_string(&self) -> String {
        let last_test_result_string = if let Some(last_test_results) = &self.last_test_results {
            format!("{}", last_test_results)
        } else {
            "".to_string()
        };

        let raw_bytes = self.get_parameters_num() * self.get_parameters_unit_size();
        let print_string: String = format!(
            "\
        Type: CPU\n\
        Graph Structure: {}\n\
        Parameters: {}\n\
        Raw Bytes: {}\n\
        Last Test Results: {}\n",
            self.graph_structure.to_string(),
            Self::format_count(self.get_parameters_num()),
            Self::format_bytes(raw_bytes),
            last_test_result_string
        );

        print_string
    }
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();
pub fn save_neural_network<P: AsRef<Path>>(nn: &NeuralNetworkCPU, path: P) -> std::io::Result<()> {
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

pub fn load_neural_network(path: &str) -> std::io::Result<NeuralNetworkCPU> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (decoded, len): (NeuralNetworkCPU, usize) =
        bincode::decode_from_slice(&buffer[..], BINCODE_CONFIG)
            .expect("load_neural_network failed, decoding failed.");

    assert_eq!(len, buffer.len()); // read all bytes
    Ok(decoded)
}

pub fn test_nn_cpu<'a>(
    nn: &'a mut NeuralNetworkCPU,
    test_data: &[DataPoint],
    is_correct_fn: ConfusionEvaluator,
    tx_test_metadata: Option<Sender<TrainingThreadPayload>>,
    tx_abort: Option<Receiver<()>>,
    pingpong: &mut NeuralNetworkPingPong,
) -> Result<&'a TestResults, anyhow::Error> {
    if test_data.len() >= 1
        && test_data.first().unwrap().inputs.len() == nn.graph_structure.input_nodes
        && test_data.first().unwrap().expected_outputs.len() == nn.graph_structure.output_nodes
    {
        log::info!("Start test_nn");

        let mut metadata = AIResultMetadata::new(DatasetUsage::Test, 0.0, 0.0, 0.0);

        let mut results = Vec::with_capacity(test_data.len());
        for i in 0..test_data.len() {
            let datapoint = &test_data[i];
            let cost = calculate_cost(
                &nn.layers,
                std::slice::from_ref(&test_data[i]),
                nn.cost_fn,
                pingpong,
                nn.is_softmax_output,
            );
            if let Some(tx_test_metadata) = &tx_test_metadata {
                let mut metadata_point = AIResultMetadata::new(
                    DatasetUsage::Test,
                    cost as LayerTypeCPU,
                    cost as LayerTypeCPU,
                    0.0,
                );

                let confusion = is_correct_fn.evaluate(&pingpong.next, &datapoint.expected_outputs);
                match confusion {
                    crate::zneural_network::is_correct::ConfusionCategory::TruePositive => {
                        metadata_point.true_positives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::TrueNegative => {
                        metadata_point.true_negatives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::FalsePositive => {
                        metadata_point.false_positives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::FalseNegative => {
                        metadata_point.false_negatives += 1
                    }
                }

                metadata.merge(&metadata_point);

                tx_test_metadata
                    .send(TrainingThreadPayload {
                        payload_index: i,
                        payload_max_index: test_data.len(),
                        training_metadata: metadata.clone(),
                    })
                    .unwrap();

                if let Some(abort) = &tx_abort {
                    if abort.try_recv().is_ok() {
                        log::info!("Abort Recieved, stopping test_nn...");
                        anyhow::bail!("Aborted")
                    }
                }
            }

            results.push((test_data[i].clone(), pingpong.next.clone()));
        }

        // TODO: Do not calculate_cost another time here
        let cost = calculate_cost(
            &nn.layers,
            test_data,
            nn.cost_fn,
            pingpong,
            nn.is_softmax_output,
        );
        let test_results = TestResults::new(results, is_correct_fn, cost);
        nn.last_test_results = Some(test_results);
        Ok(&nn.last_test_results.as_ref().unwrap())
    } else {
        anyhow::bail!("Failed to test_nn")
    }
}
