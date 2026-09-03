use candle_core::{DType, Device, Tensor};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};

use crate::datapoint::DataPoint;
use crate::neuralnetwork_cpu::GraphStructure;
use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::CostFunction;
use crate::zneural_network::is_correct::{ConfusionCategory, ConfusionEvaluator};
use crate::zneural_network::thread::TrainingThreadPayload;
use crate::zneural_network::training::{
    test_nn_gpu, AIResultMetadata, DatasetUsage, FloatDecay, TestResults,
};
use zaoai_types::ai_labels::LayerTypeCPU;

#[derive(bincode::Encode, bincode::Decode)]
pub struct SerializedLayer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
}

#[derive(bincode::Encode, bincode::Decode)]
pub struct SerializedNeuralNetwork {
    pub graph_structure: GraphStructure,
    pub layers: Vec<SerializedLayer>,
    pub last_test_results: Option<TestResults>,
    pub is_softmax_output: bool,
    pub layer_activation_function: ActivationFunctionType,
    pub cost_fn: CostFunction,
    pub version: u8,
}

#[derive(Clone)]
pub struct LayerGPU {
    pub weights: Tensor,
    pub biases: Tensor,
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
}
impl LayerGPU {
    pub fn new(in_nodes: usize, out_nodes: usize, device: &Device) -> candle_core::Result<Self> {
        let weights = Tensor::randn(0f32, 1f32, (out_nodes, in_nodes), device)?;
        let biases = Tensor::zeros((out_nodes, 1), DType::F32, device)?;

        Ok(Self {
            weights,
            biases,
            num_in_nodes: in_nodes,
            num_out_nodes: out_nodes,
        })
    }

    pub fn forward_manual(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let z = self.weights.matmul(input)?;
        let z = z.broadcast_add(&self.biases)?;
        Ok(z)
    }

    pub fn backward_manual(
        &self,
        input: &Tensor,
        d_z: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let input_t = input.t()?;
        let d_weights = d_z.matmul(&input_t)?;
        let d_biases = d_z.sum_keepdim(1)?;
        let weights_t = self.weights.t()?;
        let d_input = weights_t.matmul(d_z)?;

        Ok((d_weights, d_biases, d_input))
    }
}

pub struct LayerCacheGPU {
    pub input: Tensor,
    pub z: Tensor,
    pub output: Tensor,
}

// #[derive(bincode::Encode, bincode::Decode)]
#[derive(Clone)]
pub struct NeuralNetworkGPU {
    pub graph_structure: GraphStructure,
    pub layers: Vec<LayerGPU>,
    pub last_test_results: Option<TestResults>,
    pub is_softmax_output: bool,
    pub device: Device,
    layer_activation_function: ActivationFunctionType,
    pub cost_fn: CostFunction,
    version: u8,
}

impl NeuralNetworkGPU {
    const VERSION: u8 = 2;

    pub fn to_serializable(&self) -> candle_core::Result<SerializedNeuralNetwork> {
        let mut serializable_layers = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let weights_vec = layer.weights.flatten_all()?.to_vec1::<f32>()?;
            let biases_vec = layer.biases.flatten_all()?.to_vec1::<f32>()?;

            serializable_layers.push(SerializedLayer {
                weights: weights_vec,
                biases: biases_vec,
                num_in_nodes: layer.num_in_nodes,
                num_out_nodes: layer.num_out_nodes,
            });
        }

        Ok(SerializedNeuralNetwork {
            graph_structure: self.graph_structure.clone(),
            layers: serializable_layers,
            last_test_results: self.last_test_results.clone(),
            is_softmax_output: self.is_softmax_output,
            layer_activation_function: self.layer_activation_function.clone(),
            cost_fn: self.cost_fn.clone(),
            version: self.version,
        })
    }

    pub fn from_serializable(
        data: SerializedNeuralNetwork,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(data.layers.len());

        for layer_data in data.layers {
            let weights = Tensor::from_vec(
                layer_data.weights,
                (layer_data.num_out_nodes, layer_data.num_in_nodes),
                device,
            )?;
            let biases =
                Tensor::from_vec(layer_data.biases, (layer_data.num_out_nodes, 1), device)?;

            layers.push(LayerGPU {
                weights,
                biases,
                num_in_nodes: layer_data.num_in_nodes,
                num_out_nodes: layer_data.num_out_nodes,
            });
        }

        Ok(Self {
            graph_structure: data.graph_structure,
            layers,
            last_test_results: data.last_test_results,
            is_softmax_output: data.is_softmax_output,
            device: device.clone(),
            layer_activation_function: data.layer_activation_function,
            cost_fn: data.cost_fn,
            version: data.version,
        })
    }

    pub fn new(
        graph_structure: GraphStructure,
        layer_activation: ActivationFunctionType,
        cost_fn: CostFunction,
        _dropout_prob: Option<LayerTypeCPU>,
        device: Device,
    ) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(graph_structure.hidden_layers.len() + 1);
        let mut prev_out_size = graph_structure.input_nodes;

        for &nodes in &graph_structure.hidden_layers {
            layers.push(LayerGPU::new(prev_out_size, nodes, &device)?);
            prev_out_size = nodes;
        }

        layers.push(LayerGPU::new(
            prev_out_size,
            graph_structure.output_nodes,
            &device,
        )?);

        Ok(Self {
            graph_structure,
            layers,
            last_test_results: None,
            is_softmax_output: false,
            device,
            layer_activation_function: layer_activation,
            cost_fn,
            version: Self::VERSION,
        })
    }

    pub fn max_layer_nodes(&self) -> usize {
        let hidden_layers_max = *self
            .graph_structure
            .hidden_layers
            .iter()
            .max()
            .unwrap_or(&0);
        self.graph_structure
            .input_nodes
            .max(self.graph_structure.output_nodes)
            .max(hidden_layers_max)
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

    pub fn forward(&self, inputs: &[LayerTypeCPU]) -> candle_core::Result<Vec<LayerTypeCPU>> {
        let mut current = Tensor::from_slice(inputs, (inputs.len(), 1), &self.device)?;
        let last_idx = self.layers.len() - 1;

        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward_manual(&current)?;
            if i != last_idx {
                current = current.relu()?;
            }
        }

        let flattened = current.flatten_all()?;
        let result: Vec<f32> = flattened.to_vec1()?;
        Ok(result)
    }

    fn forward_cache(
        &self,
        inputs: &[LayerTypeCPU],
    ) -> candle_core::Result<(Vec<LayerTypeCPU>, Vec<LayerCacheGPU>)> {
        let mut current = Tensor::from_slice(inputs, (inputs.len(), 1), &self.device)?;
        let mut cache = Vec::with_capacity(self.layers.len());
        let last_idx = self.layers.len() - 1;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = current.clone();
            let z = layer.forward_manual(&layer_input)?;
            let output = if i != last_idx { z.relu()? } else { z.clone() };

            cache.push(LayerCacheGPU {
                input: layer_input,
                z,
                output: output.clone(),
            });
            current = output;
        }

        let final_output = current.flatten_all()?.to_vec1()?;
        Ok((final_output, cache))
    }

    pub fn learn_batch(
        &mut self,
        batch_data: &[DataPoint],
        learn_rate: LayerTypeCPU,
        batch_data_cost: &mut LayerTypeCPU,
        batch_data_loss: &mut LayerTypeCPU,
        _pingpong: &mut crate::neuralnetwork_cpu::NeuralNetworkPingPong,
    ) -> Vec<Vec<LayerTypeCPU>> {
        assert!(!batch_data.is_empty());

        let mut total_cost = 0.0;
        let mut last_loss = 0.0;
        let mut batch_data_outputs = Vec::with_capacity(batch_data.len());

        let mut accum_dw: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let mut accum_db: Vec<Tensor> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            accum_dw.push(layer.weights.zeros_like().unwrap());
            accum_db.push(layer.biases.zeros_like().unwrap());
        }

        for (i, datapoint) in batch_data.iter().enumerate() {
            let (predicted, cache) = self.forward_cache(&datapoint.inputs).unwrap();
            let cost = self.cost_fn.call(&predicted, &datapoint.expected_outputs);
            total_cost += cost;
            if i == batch_data.len() - 1 {
                last_loss = cost;
            }
            batch_data_outputs.push(predicted.clone());

            let last = self.layers.len() - 1;

            let mut d_z_vec = Vec::with_capacity(predicted.len());
            for (p, e) in predicted.iter().zip(datapoint.expected_outputs.iter()) {
                d_z_vec.push(p - e);
            }
            let mut d_z = Tensor::from_slice(&d_z_vec, (d_z_vec.len(), 1), &self.device).unwrap();

            for layer_idx in (0..=last).rev() {
                let layer_cache = &cache[layer_idx];
                let layer = &self.layers[layer_idx];

                let (d_w, d_b, d_input) = layer.backward_manual(&layer_cache.input, &d_z).unwrap();

                accum_dw[layer_idx] = (accum_dw[layer_idx].clone() + d_w).unwrap();
                accum_db[layer_idx] = (accum_db[layer_idx].clone() + d_b).unwrap();

                if layer_idx > 0 {
                    let prev_cache = &cache[layer_idx - 1];
                    let relu_grad = prev_cache
                        .z
                        .ge(&prev_cache.z.zeros_like().unwrap())
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap();
                    d_z = d_input.broadcast_mul(&relu_grad).unwrap();
                }
            }
        }

        let batch_len_f32 = batch_data.len() as f32;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mean_dw = (accum_dw[layer_idx].clone() / (batch_len_f32 as f64)).unwrap();
            let mean_db = (accum_db[layer_idx].clone() / (batch_len_f32 as f64)).unwrap();

            layer.weights =
                (layer.weights.clone() - (mean_dw * (learn_rate as f64)).unwrap()).unwrap();
            layer.biases =
                (layer.biases.clone() - (mean_db * (learn_rate as f64)).unwrap()).unwrap();
        }

        *batch_data_cost = total_cost / batch_len_f32;
        *batch_data_loss = last_loss;

        batch_data_outputs
    }

    pub fn learn_epoch(
        &mut self,
        _epoch_index: usize,
        training_data: &[DataPoint],
        batch_size: usize,
        learn_rate: LayerTypeCPU,
        is_correct_fn: ConfusionEvaluator,
        mut epoch_metadata: Option<&mut AIResultMetadata>,
        pingpong: &mut crate::neuralnetwork_cpu::NeuralNetworkPingPong,
    ) {
        assert!(!training_data.is_empty());
        let mut cur_index = 0;
        let len = training_data.len();

        let mut process_batch =
            |data: &[DataPoint], _batch_num: usize, _total_batches: usize, _cur_index: usize| {
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
        epoch_data_outputs: &[Vec<LayerTypeCPU>],
        epoch_data_cost: LayerTypeCPU,
        is_correct_fn: ConfusionEvaluator,
        new_metadata: &mut AIResultMetadata,
    ) {
        for (i, data) in epoch_data.iter().enumerate() {
            let datapoint_output = &epoch_data_outputs[i];
            let confusion_cat = is_correct_fn.evaluate(datapoint_output, &data.expected_outputs);
            match confusion_cat {
                ConfusionCategory::TruePositive => {
                    new_metadata.true_positives += 1;
                }
                ConfusionCategory::TrueNegative => {
                    new_metadata.true_negatives += 1;
                }
                ConfusionCategory::FalsePositive => {
                    new_metadata.false_positives += 1;
                }
                ConfusionCategory::FalseNegative => {
                    new_metadata.false_negatives += 1;
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
        assert!(!training_data.is_empty());
        assert!(batch_size > 0);

        let pingpong =
            &mut crate::neuralnetwork_cpu::NeuralNetworkPingPong::new(self.max_layer_nodes());

        for e in 0..num_epochs {
            let mut test_nn_and_send_payload =
                |tx: &Sender<TrainingThreadPayload>, data: &[DataPoint], payload_index: usize| {
                    if let Some(test_results) =
                        test_nn_gpu(self, data, is_correct_fn, None, None).ok()
                    {
                        let mut result_metadata = AIResultMetadata::from_accuracy(
                            test_results.accuracy.unwrap_or_default() as f64,
                            test_results.results.len(),
                        );
                        result_metadata.cost = test_results.cost as LayerTypeCPU;
                        result_metadata.last_loss = test_results.cost as LayerTypeCPU;
                        result_metadata.learn_rate = learn_rate;

                        let payload = TrainingThreadPayload {
                            payload_index,
                            payload_max_index: num_epochs - 1,
                            training_metadata: result_metadata,
                        };
                        if let Err(err) = tx.send(payload) {
                            log::error!(
                                "Failed to send training metadata through channel: {}",
                                err
                            );
                        }
                    }
                };

            if e == 0 {
                if let Some(tx_testing_metadata) = tx_training_metadata {
                    test_nn_and_send_payload(tx_testing_metadata, training_data, e);
                }
            }
            if validation_each_epoch != 0 && e % validation_each_epoch == 0 {
                if let Some(tx_validation_metadata) = tx_validation_metadata {
                    test_nn_and_send_payload(tx_validation_metadata, validation_data, e);
                }
            }

            let mut metadata = AIResultMetadata::new(DatasetUsage::Training, 0.0, 0.0, 0.0);

            let maybe_decayed_learn_rate = learn_rate_decay
                .as_ref()
                .map(|f| f.decay(learn_rate, e))
                .unwrap_or(learn_rate);

            self.learn_epoch(
                e,
                training_data,
                batch_size,
                maybe_decayed_learn_rate,
                is_correct_fn,
                Some(&mut metadata),
                pingpong,
            );

            if let Some(tx) = tx_training_metadata {
                let payload = TrainingThreadPayload {
                    payload_index: e + 1,
                    payload_max_index: num_epochs - 1,
                    training_metadata: metadata,
                };
                if let Err(err) = tx.send(payload) {
                    log::error!("Failed to send training metadata through channel: {}", err);
                }
            }

            if let Some(post_fn) = &eval_abort_fn {
                if post_fn() {
                    break;
                }
            }
        }
    }

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

    pub fn to_string(&self) -> String {
        let last_test_result_string = if let Some(res) = &self.last_test_results {
            format!("{}", res)
        } else {
            "".to_string()
        };
        format!(
            "\
Graph Structure: {}\n\
Parameters: {}\n\
Raw Bytes: {}\n\
Last Test Results: {}\n",
            self.graph_structure.to_string(),
            Self::format_count(self.get_parameters_num()),
            self.get_parameters_num() * self.get_parameters_unit_size(),
            last_test_result_string
        )
    }
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();

pub fn save_neural_network<P: AsRef<Path>>(nn: &NeuralNetworkGPU, path: P) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let serializable = nn
        .to_serializable()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let encoded: Vec<u8> = bincode::encode_to_vec(&serializable, BINCODE_CONFIG)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}
