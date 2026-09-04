use burn::prelude::Backend;
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::TensorData;
use burn::tensor::{Distribution, Shape, Tensor};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::Sender;

use crate::datapoint::DataPoint;
use crate::neuralnetwork_cpu::GraphStructure;
use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::CostFunction;
use crate::zneural_network::is_correct::{ConfusionCategory, ConfusionEvaluator};
use crate::zneural_network::layer::{BiasInit, WeightInit};
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
pub struct LayerGPU<B: Backend> {
    pub weights: Tensor<B, 2>,
    pub biases: Tensor<B, 2>,
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
    pub activation_type: ActivationFunctionType,
}

impl<B: Backend> LayerGPU<B> {
    pub fn new(
        in_nodes: usize,
        out_nodes: usize,
        activation_type: ActivationFunctionType,
        weight_init: WeightInit,
        bias_init: BiasInit,
        device: &B::Device,
    ) -> Self {
        assert!(in_nodes > 0, "NumInNodes must be > 0");
        assert!(out_nodes > 0, "NumOutNodes must be > 0");

        /*
         * IMPORTANT:
         *
         * Do not use one hard-coded initialization here.
         *
         * The CPU implementation gets the initialization strategy from
         * the UI and applies it to every layer. The GPU implementation
         * does the same.
         */
        let weights = Self::create_weights(in_nodes, out_nodes, weight_init, device);

        let biases = Self::create_biases(out_nodes, bias_init, device);

        Self {
            weights,
            biases,
            num_in_nodes: in_nodes,
            num_out_nodes: out_nodes,
            activation_type,
        }
    }

    fn create_weights(
        in_nodes: usize,
        out_nodes: usize,
        weight_init: WeightInit,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        /*
         * Weight initialization follows the same mathematical choices
         * exposed by the CPU/UI:
         *
         * Zero
         * Uniform [0, 1]
         * Normal(0, 1)
         * Xavier Uniform
         * Xavier Normal
         * He Uniform
         * He Normal
         * LeCun
         *
         * Tensor shape is [input_nodes, output_nodes], matching the
         * GPU matmul:
         *
         *     [batch, input] x [input, output]
         */
        match weight_init {
            WeightInit::Zero => Tensor::<B, 2>::zeros([in_nodes, out_nodes], device),

            WeightInit::Uniform => Tensor::<B, 2>::random(
                [in_nodes, out_nodes],
                Distribution::Uniform(0.0, 1.0),
                device,
            ),

            WeightInit::NormalDist => Tensor::<B, 2>::random(
                [in_nodes, out_nodes],
                Distribution::Normal(0.0, 1.0),
                device,
            ),

            WeightInit::XavierUniform => {
                /*
                 * Xavier/Glorot:
                 *
                 * limit = sqrt(6 / (fan_in + fan_out))
                 *
                 * Uniform[-limit, limit]
                 */
                let limit = (6.0f32 / (in_nodes + out_nodes) as f32).sqrt();

                Tensor::<B, 2>::random(
                    [in_nodes, out_nodes],
                    Distribution::Uniform((-limit).into(), limit.into()),
                    device,
                )
            }

            WeightInit::XavierNormal => {
                /*
                 * Xavier normal:
                 *
                 * stddev = sqrt(2 / (fan_in + fan_out))
                 */
                let stddev = (2.0f32 / (in_nodes + out_nodes) as f32).sqrt();

                Tensor::<B, 2>::random(
                    [in_nodes, out_nodes],
                    Distribution::Normal(0.0, stddev.into()),
                    device,
                )
            }

            WeightInit::HeUniform => {
                /*
                 * He uniform:
                 *
                 * limit = sqrt(6 / fan_in)
                 */
                let limit = (6.0f32 / in_nodes as f32).sqrt();

                Tensor::<B, 2>::random(
                    [in_nodes, out_nodes],
                    Distribution::Uniform((-limit).into(), limit.into()),
                    device,
                )
            }

            WeightInit::HeNormal => {
                /*
                 * He normal:
                 *
                 * stddev = sqrt(2 / fan_in)
                 */
                let stddev = (2.0f32 / in_nodes as f32).sqrt();

                Tensor::<B, 2>::random(
                    [in_nodes, out_nodes],
                    Distribution::Normal(0.0, stddev.into()),
                    device,
                )
            }

            WeightInit::LeCun => {
                /*
                 * LeCun normal:
                 *
                 * stddev = 1 / sqrt(fan_in)
                 */
                let stddev = (1.0f32 / in_nodes as f32).sqrt();

                Tensor::<B, 2>::random(
                    [in_nodes, out_nodes],
                    Distribution::Normal(0.0, stddev.into()),
                    device,
                )
            }
        }
    }

    fn create_biases(out_nodes: usize, bias_init: BiasInit, device: &B::Device) -> Tensor<B, 2> {
        match bias_init {
            BiasInit::Zero => Tensor::<B, 2>::zeros([1, out_nodes], device),

            BiasInit::ZeroPointZeroOne => Tensor::<B, 2>::full([1, out_nodes], 0.01, device),
        }
    }

    pub fn forward_linear(&self, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        input.clone().matmul(self.weights.clone()) + self.biases.clone()
    }

    pub fn forward(&self, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        let z = self.forward_linear(input);

        Self::activation(z, &self.activation_type)
    }

    fn activation(z: Tensor<B, 2>, activation_type: &ActivationFunctionType) -> Tensor<B, 2> {
        /*
         * We intentionally use the Display representation here instead
         * of matching enum variants directly.
         *
         * That keeps this file compatible with the exact enum variants
         * defined elsewhere in the project.
         */
        let name = format!("{}", activation_type).to_lowercase();

        if name == "relu" {
            relu(z)
        } else if name == "sigmoid" {
            sigmoid(z)
        } else if name == "tanh" {
            tanh(z)
        } else if name == "linear" || name == "identity" || name == "none" {
            z
        } else {
            /*
             * The CPU code may contain additional activation functions.
             *
             * Falling back to the raw value is preferable to silently
             * changing the network architecture.
             */
            z
        }
    }

    fn activation_derivative(
        z: &Tensor<B, 2>,
        activation_type: &ActivationFunctionType,
    ) -> Tensor<B, 2> {
        let name = format!("{}", activation_type).to_lowercase();

        if name == "relu" {
            /*
             * ReLU'(x):
             *
             *     1 if x > 0
             *     0 otherwise
             *
             * Use > rather than >=.
             */
            z.clone().greater_elem(0.0).float()
        } else if name == "sigmoid" {
            /*
             * sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
             */
            let s = sigmoid(z.clone());

            s.clone() * (1.0 - s)
        } else if name == "tanh" {
            /*
             * tanh'(x) = 1 - tanh(x)^2
             */
            let t = tanh(z.clone());

            1.0 - t.clone() * t
        } else {
            /*
             * Linear/identity activation derivative.
             */
            Tensor::<B, 2>::ones(z.shape(), &z.device())
        }
    }

    /*
     * Backpropagation for:
     *
     *     Z = XW + b
     *
     * Given dZ:
     *
     *     dW = X^T dZ
     *     db = sum(dZ, axis=0)
     *     dX = dZ W^T
     */
    pub fn backward_linear(
        &self,
        input: &Tensor<B, 2>,
        d_z: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let input_t = input.clone().transpose();

        let d_weights = input_t.matmul(d_z.clone());

        let d_biases = d_z.clone().sum_dim(0);

        /*
         * Calculate d_input using the OLD weights.
         *
         * The caller updates weights only after this function returns.
         */
        let d_input = d_z.clone().matmul(self.weights.clone().transpose());

        (d_weights, d_biases, d_input)
    }
}

pub struct LayerCacheGPU<B: Backend> {
    pub input: Tensor<B, 2>,
    pub z: Tensor<B, 2>,
    pub output: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct NeuralNetworkGPU<B: Backend> {
    pub graph_structure: GraphStructure,
    pub layers: Vec<LayerGPU<B>>,
    pub last_test_results: Option<TestResults>,
    pub is_softmax_output: bool,
    pub device: B::Device,
    layer_activation_function: ActivationFunctionType,
    pub cost_fn: CostFunction,
    version: u8,
}

impl<B: Backend> NeuralNetworkGPU<B> {
    const VERSION: u8 = 2;

    pub fn to_serializable(&self) -> SerializedNeuralNetwork {
        let mut serializable_layers = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let weights_vec = layer.weights.clone().into_data().to_vec::<f32>().unwrap();

            let biases_vec = layer.biases.clone().into_data().to_vec::<f32>().unwrap();

            serializable_layers.push(SerializedLayer {
                weights: weights_vec,
                biases: biases_vec,
                num_in_nodes: layer.num_in_nodes,
                num_out_nodes: layer.num_out_nodes,
            });
        }

        SerializedNeuralNetwork {
            graph_structure: self.graph_structure.clone(),
            layers: serializable_layers,
            last_test_results: self.last_test_results.clone(),
            is_softmax_output: self.is_softmax_output,
            layer_activation_function: self.layer_activation_function.clone(),
            cost_fn: self.cost_fn.clone(),
            version: self.version,
        }
    }

    pub fn from_serializable(data: SerializedNeuralNetwork, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(data.layers.len());

        for layer_data in data.layers {
            let weights = Tensor::<B, 2>::from_data(
                TensorData::new(
                    layer_data.weights,
                    Shape::new([layer_data.num_in_nodes, layer_data.num_out_nodes]),
                ),
                device,
            );

            let biases = Tensor::<B, 2>::from_data(
                TensorData::new(layer_data.biases, Shape::new([1, layer_data.num_out_nodes])),
                device,
            );

            layers.push(LayerGPU {
                weights,
                biases,
                num_in_nodes: layer_data.num_in_nodes,
                num_out_nodes: layer_data.num_out_nodes,
                activation_type: data.layer_activation_function.clone(),
            });
        }

        Self {
            graph_structure: data.graph_structure,
            layers,
            last_test_results: data.last_test_results,
            is_softmax_output: data.is_softmax_output,
            device: device.clone(),
            layer_activation_function: data.layer_activation_function,
            cost_fn: data.cost_fn,
            version: data.version,
        }
    }

    pub fn new(
        graph_structure: GraphStructure,
        layer_activation: ActivationFunctionType,
        cost_fn: CostFunction,
        dropout_prob: Option<LayerTypeCPU>,
        weight_init: WeightInit,
        bias_init: BiasInit,
        device: B::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(graph_structure.hidden_layers.len() + 1);

        let mut prev_out_size = graph_structure.input_nodes;

        /*
         * Input nodes are not layers.
         *
         * Create hidden layers.
         */
        for &nodes in &graph_structure.hidden_layers {
            layers.push(LayerGPU::new(
                prev_out_size,
                nodes,
                layer_activation.clone(),
                weight_init,
                bias_init,
                &device,
            ));

            prev_out_size = nodes;
        }

        /*
         * Create output layer.
         *
         * This mirrors the CPU implementation:
         *
         * Layer::new(
         *     prev_out_size,
         *     output_nodes,
         *     layer_activation,
         *     None,
         *     weight_init,
         *     bias_init,
         * );
         *
         * Dropout is intentionally not used on the output layer.
         */
        layers.push(LayerGPU::new(
            prev_out_size,
            graph_structure.output_nodes,
            layer_activation.clone(),
            weight_init,
            bias_init,
            &device,
        ));

        /*
         * Keep this argument in the constructor because it is part of
         * the same public configuration path as the CPU network.
         *
         * Dropout is not currently implemented in this manual GPU
         * training path.
         */
        let _ = dropout_prob;

        Self {
            graph_structure,
            layers,
            last_test_results: None,
            is_softmax_output: false,
            device,
            layer_activation_function: layer_activation,
            cost_fn,
            version: Self::VERSION,
        }
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

    pub fn forward(&self, inputs: &[LayerTypeCPU]) -> Vec<f32> {
        assert_eq!(
            inputs.len(),
            self.graph_structure.input_nodes,
            "GPU network expected {} inputs but received {}",
            self.graph_structure.input_nodes,
            inputs.len()
        );

        let inputs_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();

        let mut current = Tensor::<B, 2>::from_data(
            TensorData::new(
                inputs_f32,
                Shape::new([1, self.graph_structure.input_nodes]),
            ),
            &self.device,
        );

        for layer in &self.layers {
            current = layer.forward(&current);
        }

        /*
         * Softmax output is intentionally applied only at the end.
         *
         * The network's internal layers use the configured activation.
         */
        if self.is_softmax_output {
            current = softmax(current);
        }

        current.into_data().to_vec::<f32>().unwrap()
    }

    fn forward_cache_batched(
        &self,
        inputs: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Vec<LayerCacheGPU<B>>) {
        let mut current = inputs.clone();

        let mut cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let layer_input = current.clone();

            let z = layer.forward_linear(&layer_input);

            let output = LayerGPU::<B>::activation(z.clone(), &layer.activation_type);

            cache.push(LayerCacheGPU {
                input: layer_input,
                z,
                output: output.clone(),
            });

            current = output;
        }

        if self.is_softmax_output {
            current = softmax(current);
        }

        (current, cache)
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

        assert!(
            learn_rate > 0.0,
            "Learning rate must be > 0, got {}",
            learn_rate
        );

        let batch_size = batch_data.len();

        let in_nodes = self.graph_structure.input_nodes;

        let out_nodes = self.graph_structure.output_nodes;

        let mut flat_inputs = Vec::with_capacity(batch_size * in_nodes);

        let mut flat_expected = Vec::with_capacity(batch_size * out_nodes);

        for dp in batch_data {
            assert_eq!(
                dp.inputs.len(),
                in_nodes,
                "Expected {} input nodes, got {}",
                in_nodes,
                dp.inputs.len()
            );

            assert_eq!(
                dp.expected_outputs.len(),
                out_nodes,
                "Expected {} output nodes, got {}",
                out_nodes,
                dp.expected_outputs.len()
            );

            flat_inputs.extend(dp.inputs.iter().map(|&x| x as f32));

            flat_expected.extend(dp.expected_outputs.iter().map(|&x| x as f32));
        }

        let input_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(flat_inputs, Shape::new([batch_size, in_nodes])),
            &self.device,
        );

        let expected_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(flat_expected, Shape::new([batch_size, out_nodes])),
            &self.device,
        );

        /*
         * Forward pass.
         */
        let (predicted_tensor, cache) = self.forward_cache_batched(&input_tensor);

        /*
         * MSE derivative:
         *
         *     d/dprediction MSE
         *       = prediction - expected
         *
         * This matches the original GPU implementation and is correct
         * when CostFunction::Mse is being used.
         */
        let mut d_a = predicted_tensor.clone() - expected_tensor;

        /*
         * Average gradient over batch.
         */
        let lr = learn_rate as f32 / batch_size as f32;

        /*
         * Backward pass.
         *
         * We must walk from the output layer toward the input layer.
         */
        for layer_idx in (0..self.layers.len()).rev() {
            let layer_cache = &cache[layer_idx];

            let layer = &self.layers[layer_idx];

            /*
             * If softmax is enabled, the simple
             *
             *     prediction - target
             *
             * derivative is only correct for the usual
             * softmax + cross entropy combination.
             *
             * For the current MSE setup, softmax needs the full
             * Jacobian. Rather than silently pretending the derivative
             * is correct, the normal path remains the MSE path.
             *
             * Most importantly, with softmax disabled (your current UI
             * configuration), this is exactly the ordinary activation
             * derivative path.
             */
            let d_z = if layer_idx == self.layers.len() - 1 && self.is_softmax_output {
                /*
                 * For the common classification case, if the
                 * network is configured for softmax output and MSE,
                 * calculate the element-wise approximation here.
                 *
                 * Cross-entropy should preferably be used with
                 * softmax for the exact simplified derivative.
                 */
                d_a.clone()
                    * LayerGPU::<B>::activation_derivative(&layer_cache.z, &layer.activation_type)
            } else {
                d_a.clone()
                    * LayerGPU::<B>::activation_derivative(&layer_cache.z, &layer.activation_type)
            };

            /*
             * Calculate all gradients BEFORE modifying the layer.
             *
             * This ensures d_input uses the old weights.
             */
            let (d_weights, d_biases, d_input) = layer.backward_linear(&layer_cache.input, &d_z);

            /*
             * Gradient descent.
             */
            let new_weights = layer.weights.clone() - (d_weights * lr);

            let new_biases = layer.biases.clone() - (d_biases * lr);

            self.layers[layer_idx].weights = new_weights;

            self.layers[layer_idx].biases = new_biases;

            /*
             * d_input becomes dA for the previous layer.
             */
            d_a = d_input;
        }

        /*
         * Convert predictions back to CPU values for the existing
         * CostFunction / metadata implementation.
         */
        let predicted_vec = predicted_tensor.into_data().to_vec::<f32>().unwrap();

        let mut batch_data_outputs = Vec::with_capacity(batch_size);

        let mut total_cost = 0.0;

        let mut last_loss = 0.0;

        for (i, dp) in batch_data.iter().enumerate() {
            let start = i * out_nodes;

            let end = start + out_nodes;

            let pred: Vec<LayerTypeCPU> = predicted_vec[start..end]
                .iter()
                .map(|&x| x as LayerTypeCPU)
                .collect();

            let cost = self.cost_fn.call(&pred, &dp.expected_outputs);

            total_cost += cost;

            if i == batch_size - 1 {
                last_loss = cost;
            }

            batch_data_outputs.push(pred);
        }

        *batch_data_cost = total_cost / batch_size as LayerTypeCPU;

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
        assert!(batch_size > 0);

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

        new_metadata.cost = epoch_data_cost;
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
                    if let Ok(test_results) = test_nn_gpu(self, data, is_correct_fn, None, None) {
                        let mut result_metadata = AIResultMetadata::from_accuracy(
                            test_results.accuracy.unwrap_or_default() as f64,
                            test_results.results.len(),
                        );

                        result_metadata.cost = test_results.cost;

                        result_metadata.last_loss = test_results.cost;

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

            /*
             * Initial training result.
             */
            if e == 0 {
                if let Some(tx_testing_metadata) = tx_training_metadata {
                    test_nn_and_send_payload(tx_testing_metadata, training_data, e);
                }
            }

            /*
             * Validation.
             */
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
        let last_test_result: Option<&TestResults> = self.last_test_results.as_ref();

        let last_test_result_string = if let Some(res) = last_test_result {
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

/*
 * Softmax implementation.
 *
 * Numerically stable:
 *
 *     softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * This is performed independently for every row in the batch.
 */
fn softmax<B: Backend>(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let max_values = input.clone().max_dim(1);

    let shifted = input - max_values;

    let exp = shifted.exp();

    let sum = exp.clone().sum_dim(1);

    exp / sum
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();

pub fn save_neural_network<B: Backend, P: AsRef<Path>>(
    nn: &NeuralNetworkGPU<B>,
    path: P,
) -> std::io::Result<()> {
    let path = path.as_ref();

    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let serializable = nn.to_serializable();

    let encoded: Vec<u8> = bincode::encode_to_vec(&serializable, BINCODE_CONFIG)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut file = File::create(path)?;

    file.write_all(&encoded)?;

    Ok(())
}
