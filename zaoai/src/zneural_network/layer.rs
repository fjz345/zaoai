use rand::prelude::*;
use rand_chacha::{self, ChaCha8Rng};
use rand_distr::{num_traits::FromPrimitive, Distribution, Normal};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use wide::f32x8;

use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::CostFunction;
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::neuralnetwork::NeuralNetworkPingPong;
use zaoai_types::ai_labels::LayerTypeCPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct Layer {
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
    pub weights: Vec<Vec<LayerTypeCPU>>,
    pub biases: Vec<LayerTypeCPU>,
    pub weights_cost_grads: Vec<Vec<LayerTypeCPU>>,
    pub biases_cost_grads: Vec<LayerTypeCPU>,
    pub activation_type: ActivationFunctionType,
    pub dropout_prob: Option<LayerTypeCPU>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct LayerLearnData {
    pub inputs: Vec<LayerTypeCPU>,
    pub weighted_inputs: Vec<LayerTypeCPU>,
    pub activation_values: Vec<LayerTypeCPU>,
    //"node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub node_values: Vec<LayerTypeCPU>,
    pub dropout_mask: Option<Vec<LayerTypeCPU>>, // same length as layer outputs
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum WeightInit {
    Zero,       // Bad
    Uniform,    // Uniform [0, 1]
    NormalDist, // Normal(0, 1)
    #[default]
    XavierUniform, // sigmoid / tanh
    XavierNormal, // sigmoid / tanh
    HeUniform,  // ReLU / leaky ReLU
    HeNormal,   // ReLU / leaky ReLU
    LeCun,      // SELU / scaled tanh
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum BiasInit {
    Zero,
    #[default]
    ZeroPointZeroOne,
    // Random,
}

pub struct WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: Distribution<T>,
{
    pub weight_init: WeightInit,
    #[allow(clippy::collection_is_never_read)]
    pub num_inputs: usize,
    #[allow(clippy::collection_is_never_read)]
    pub num_outputs: usize,
    pub normal_dist: Option<rand_distr::Normal<T>>,
    pub limit: Option<T>,
}

impl<T> WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: rand::distributions::Distribution<T>,
{
    #[inline(always)]
    fn to_t(x: f64) -> T {
        T::from_f64(x).expect("conversion from f64 to T failed")
    }

    pub fn new(weight_init: WeightInit, num_inputs: usize, num_outputs: usize) -> Self {
        let (normal_dist, limit) = match weight_init {
            WeightInit::NormalDist => (
                Some(Normal::new(Self::to_t(0.0), Self::to_t(1.0)).unwrap()),
                None,
            ),
            WeightInit::XavierUniform => {
                let limit = (Self::to_t(6.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (None, Some(limit))
            }
            WeightInit::XavierNormal => {
                let std_dev = (Self::to_t(2.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::HeUniform => {
                let limit = (Self::to_t(6.0) / Self::to_t(num_inputs as f64)).sqrt();
                (None, Some(limit))
            }
            WeightInit::HeNormal => {
                let std_dev = (Self::to_t(2.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::LeCun => {
                let std_dev = (Self::to_t(1.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            _ => (None, None), // Zero and Uniform don't need precalc
        };

        Self {
            weight_init,
            num_inputs,
            num_outputs,
            normal_dist,
            limit,
        }
    }

    pub fn sample_weight(&self, rng: &mut ChaCha8Rng) -> T {
        match self.weight_init {
            WeightInit::Zero => Self::to_t(0.0),
            WeightInit::Uniform => {
                T::from_f64(rng.gen_range(0.0..1.0)).expect("Uniform range failed")
            }
            WeightInit::NormalDist
            | WeightInit::XavierNormal
            | WeightInit::HeNormal
            | WeightInit::LeCun => self.normal_dist.as_ref().unwrap().sample(rng),
            WeightInit::XavierUniform | WeightInit::HeUniform => {
                let limit = self.limit.unwrap();
                let val = rng.gen_range(-limit.to_f64().unwrap()..limit.to_f64().unwrap());
                T::from_f64(val).unwrap()
            }
        }
    }
}

impl WeightInit {
    pub fn all() -> &'static [Self] {
        use crate::WeightInit::*;
        &[
            Zero,
            Uniform,
            NormalDist,
            XavierUniform,
            XavierNormal,
            HeUniform,
            HeNormal,
            LeCun,
        ]
    }
}

impl BiasInit {
    pub fn all() -> &'static [Self] {
        use crate::BiasInit::*;
        &[Zero, ZeroPointZeroOne]
    }

    pub fn sample_bias(self) -> LayerTypeCPU {
        match self {
            Self::Zero => 0.0,
            Self::ZeroPointZeroOne => 0.01,
        }
    }
}

impl Layer {
    pub fn new(
        num_in_nodes: usize,
        num_out_nodes: usize,
        activation_type: ActivationFunctionType,
        dropout_prob: Option<LayerTypeCPU>,
        weight_init: WeightInit,
        bias_init: BiasInit,
    ) -> Self {
        assert!(num_in_nodes > 0, "NumInNodes must be > 0");
        assert!(num_out_nodes > 0, "NumOutNodes must be > 0");

        // Initialize weights and gradients with zeros
        let weights = vec![vec![0.0; num_in_nodes]; num_out_nodes];
        let weights_cost_grads = vec![vec![0.0; num_in_nodes]; num_out_nodes];
        let biases = vec![0.0; num_out_nodes];
        let biases_cost_grads = vec![0.0; num_out_nodes];

        let mut layer = Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
            activation_type,
            dropout_prob,
        };

        layer.init_weights_and_biases(0, weight_init, bias_init);

        layer
    }

    pub fn init_weights_and_biases(
        &mut self,
        seed: u64,
        weight_init: WeightInit,
        bias_init: BiasInit,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let ctx = WeightInitContext::<LayerTypeCPU>::new(
            weight_init,
            self.num_in_nodes,
            self.num_out_nodes,
        );

        for i in 0..self.num_out_nodes {
            self.biases[i] = bias_init.sample_bias();

            for j in 0..self.num_in_nodes {
                self.weights[i][j] = ctx.sample_weight(&mut rng);
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    fn compute_weighted_inputs_scalar(
        &self,
        inputs: &[LayerTypeCPU],
        output_buf: &mut [LayerTypeCPU],
    ) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        for ((output, weights_row), &bias) in output_buf
            .iter_mut()
            .zip(self.weights.iter())
            .zip(self.biases.iter())
        {
            *output = bias
                + inputs
                    .iter()
                    .zip(weights_row.iter())
                    .map(|(input, weight)| input * weight)
                    .sum::<LayerTypeCPU>();
        }
    }

    #[cfg(feature = "simd")]
    fn compute_weighted_inputs_simd(&self, inputs: &[f32], output_buf: &mut [f32]) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        for ((output, weights_row), &bias) in output_buf
            .iter_mut()
            .zip(self.weights.iter())
            .zip(self.biases.iter())
        {
            let mut sum = f32x8::splat(0.0);
            let mut input_chunks = inputs.as_chunks::<8>();
            let mut weight_chunks = weights_row.as_chunks::<8>();

            for (i_chunk, w_chunk) in input_chunks.0.iter().zip(weight_chunks.0.iter()) {
                sum += f32x8::from(*i_chunk) * f32x8::from(*w_chunk);
            }

            let mut weighted_sum = sum.reduce_add();

            for (i, w) in input_chunks.1.iter().zip(weight_chunks.1.iter()) {
                weighted_sum += i * w;
            }

            *output = weighted_sum + bias;
        }
    }

    pub fn apply_activation(weighted_inputs: &mut [LayerTypeCPU], t: ActivationFunctionType) {
        #[cfg(feature = "simd")]
        Self::apply_activation_simd(weighted_inputs, t);
        #[cfg(not(feature = "simd"))]
        Self::apply_activation_scalar(weighted_inputs, t);
    }

    #[cfg(not(feature = "simd"))]
    pub fn apply_activation_scalar(
        weighted_inputs: &mut [LayerTypeCPU],
        t: ActivationFunctionType,
    ) {
        weighted_inputs.iter_mut().for_each(|x| *x = t.activate(*x));
    }
    #[cfg(feature = "simd")]
    pub fn apply_activation_simd(input: &mut [f32], t: ActivationFunctionType) {
        const CHUNK_SIZE: usize = 8;

        let mut chunks = input.chunks_exact_mut(CHUNK_SIZE);
        // let mut chunks = input.as_chunks_mut::<CHUNK_SIZE>().0;

        for chunk in chunks.by_ref() {
            let arr: [f32; CHUNK_SIZE] = (&*chunk).try_into().unwrap();
            let input_vec = f32x8::from(arr);

            let activated_vec = t.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();

            chunk.copy_from_slice(&out);
        }

        for x in chunks.into_remainder() {
            *x = t.activate(*x);
        }
    }

    fn fill_learn_data(&self, learn_data: &mut LayerLearnData, weighted_inputs: &[LayerTypeCPU]) {
        #[cfg(feature = "simd")]
        self.fill_learn_data_simd(learn_data, weighted_inputs);
        #[cfg(not(feature = "simd"))]
        self.fill_learn_data_scalar(learn_data, weighted_inputs);
    }
    #[cfg(not(feature = "simd"))]
    fn fill_learn_data_scalar(
        &self,
        learn_data: &mut LayerLearnData,
        weighted_inputs: &[LayerTypeCPU],
    ) {
        assert_eq!(learn_data.weighted_inputs.len(), self.num_out_nodes);
        assert_eq!(learn_data.activation_values.len(), self.num_out_nodes);

        learn_data.weighted_inputs.copy_from_slice(weighted_inputs);

        for (w_in, act) in learn_data
            .weighted_inputs
            .iter()
            .zip(learn_data.activation_values.iter_mut())
        {
            *act = self.activation_type.activate(*w_in);
        }
    }
    #[cfg(feature = "simd")]
    fn fill_learn_data_simd(&self, learn_data: &mut LayerLearnData, weighted_inputs: &[f32]) {
        use wide::f32x8;

        assert_eq!(learn_data.weighted_inputs.len(), self.num_out_nodes);
        assert_eq!(learn_data.activation_values.len(), self.num_out_nodes);

        learn_data.weighted_inputs.copy_from_slice(weighted_inputs);

        const CHUNK_SIZE: usize = 8;

        let len = learn_data.weighted_inputs.len();
        let chunks = learn_data.weighted_inputs.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        learn_data.activation_values.clear();
        learn_data.activation_values.reserve(len);

        for chunk in chunks {
            let input_vec = f32x8::from(chunk);
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            learn_data.activation_values.extend_from_slice(&out);
        }

        if !remainder.is_empty() {
            let mut padded = [0.0f32; CHUNK_SIZE];
            padded[..remainder.len()].copy_from_slice(remainder);
            let input_vec = f32x8::from(padded);
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            // Copy only the valid elements (remainder.len())
            learn_data
                .activation_values
                .extend_from_slice(&out[..remainder.len()]);
        }
    }

    pub fn calculate_outputs(&self, inputs: &[LayerTypeCPU], outputs: &mut [LayerTypeCPU]) {
        #[cfg(not(feature = "simd"))]
        self.calculate_outputs_scalar(inputs, outputs);
        #[cfg(feature = "simd")]
        self.calculate_outputs_simd(inputs, outputs);
    }
    #[cfg(not(feature = "simd"))]
    pub fn calculate_outputs_scalar(&self, inputs: &[LayerTypeCPU], outputs: &mut [LayerTypeCPU]) {
        self.compute_weighted_inputs_scalar(inputs, outputs);
        Self::apply_activation(outputs, self.activation_type);
    }
    #[cfg(feature = "simd")]
    pub fn calculate_outputs_simd(&self, inputs: &[f32], outputs: &mut [f32]) {
        self.compute_weighted_inputs_simd(inputs, outputs);
        Self::apply_activation(outputs, self.activation_type);
    }

    pub fn calculate_outputs_learn(
        &mut self,
        inputs: &[LayerTypeCPU],
        outputs: &mut [LayerTypeCPU],
        learn_data: &mut LayerLearnData,
    ) {
        #[cfg(feature = "simd")]
        self.calculate_outputs_learn_simd(inputs, outputs, learn_data);
        #[cfg(not(feature = "simd"))]
        self.calculate_outputs_learn_scalar(inputs, outputs, learn_data);
    }
    #[cfg(not(feature = "simd"))]
    pub fn calculate_outputs_learn_scalar(
        &mut self,
        inputs: &[LayerTypeCPU],
        outputs: &mut [LayerTypeCPU],
        learn_data: &mut LayerLearnData,
    ) {
        learn_data.inputs.clear();
        learn_data.inputs.extend_from_slice(inputs);

        self.compute_weighted_inputs_scalar(inputs, outputs);
        self.fill_learn_data(learn_data, &outputs);
        outputs.copy_from_slice(&learn_data.activation_values);
    }
    #[cfg(feature = "simd")]
    pub fn calculate_outputs_learn_simd(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        learn_data: &mut LayerLearnData,
    ) {
        learn_data.inputs.clear();
        learn_data.inputs.extend_from_slice(inputs);

        self.compute_weighted_inputs_simd(inputs, outputs);
        self.fill_learn_data(learn_data, &outputs);
        outputs.copy_from_slice(&learn_data.activation_values);
    }

    pub fn apply_cost_gradient(&mut self, learn_rate: LayerTypeCPU) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    #[inline]
    pub fn update_cost_gradients(&mut self, learn_data: &mut LayerLearnData) {
        #[cfg(feature = "simd")]
        self.update_cost_gradients_simd(learn_data);
        #[cfg(not(feature = "simd"))]
        self.update_cost_gradients_scalar(learn_data);
    }

    fn update_cost_gradient_for_node(
        weight_grad_row: &mut [LayerTypeCPU],
        bias_grad: &mut LayerTypeCPU,
        node_value: LayerTypeCPU,
        inputs: &[LayerTypeCPU],
        num_in_nodes: usize,
    ) {
        for node_in in 0..num_in_nodes {
            let derivative_cost_weight = inputs[node_in] * node_value;
            weight_grad_row[node_in] += derivative_cost_weight;
        }
        *bias_grad += node_value; // same as 1.0 * node_value
    }
    #[cfg(feature = "simd")]
    fn update_cost_gradient_for_node_simd(
        weight_grad_row: &mut [f32],
        bias_grad: &mut f32,
        node_value: f32,
        inputs: &[f32],
        num_in_nodes: usize,
    ) {
        const CHUNK_SIZE: usize = 8;
        let chunks = num_in_nodes / CHUNK_SIZE;
        let remainder = num_in_nodes % CHUNK_SIZE;
        let node_value_vec = f32x8::splat(node_value);

        for i in 0..chunks {
            let offset = i * CHUNK_SIZE;
            let input_vec = f32x8::from(&inputs[offset..offset + CHUNK_SIZE]);
            let mut grad_vec = f32x8::from(&weight_grad_row[offset..offset + CHUNK_SIZE]);
            grad_vec += input_vec * node_value_vec;
            weight_grad_row[offset..offset + CHUNK_SIZE].copy_from_slice(&grad_vec.to_array());
        }

        for i in (num_in_nodes - remainder)..num_in_nodes {
            weight_grad_row[i] += inputs[i] * node_value;
        }
        // Update bias gradient
        *bias_grad += node_value;
    }

    #[cfg(not(feature = "simd"))]
    pub fn update_cost_gradients_scalar(&mut self, learn_data: &LayerLearnData) {
        let inputs = &learn_data.inputs;
        let num_in_nodes = self.num_in_nodes;

        if let Some(mask) = learn_data.dropout_mask.as_ref() {
            for (((weight_grad_row, bias_grad), &node_value), &m) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
                .zip(mask.iter())
            {
                if m != 0.0 {
                    Self::update_cost_gradient_for_node(
                        weight_grad_row,
                        bias_grad,
                        node_value,
                        inputs,
                        num_in_nodes,
                    );
                }
            }
        } else {
            for ((weight_grad_row, bias_grad), &node_value) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
            {
                Self::update_cost_gradient_for_node(
                    weight_grad_row,
                    bias_grad,
                    node_value,
                    inputs,
                    num_in_nodes,
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    pub fn update_cost_gradients_simd(&mut self, learn_data: &LayerLearnData) {
        let inputs = &learn_data.inputs;
        let num_in_nodes = self.num_in_nodes;

        if let Some(mask) = learn_data.dropout_mask.as_ref() {
            for (((weight_grad_row, bias_grad), &node_value), &m) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
                .zip(mask.iter())
            {
                if m != 0.0 {
                    Self::update_cost_gradient_for_node_simd(
                        weight_grad_row,
                        bias_grad,
                        node_value,
                        inputs,
                        num_in_nodes,
                    );
                }
            }
        } else {
            for ((weight_grad_row, bias_grad), &node_value) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
            {
                Self::update_cost_gradient_for_node_simd(
                    weight_grad_row,
                    bias_grad,
                    node_value,
                    inputs,
                    num_in_nodes,
                );
            }
        }
    }

    pub fn clear_cost_gradient(&mut self) {
        self.biases_cost_grads.fill(0.0);
        for row in &mut self.weights_cost_grads {
            row.fill(0.0);
        }
    }

    #[cfg(not(feature = "simd"))]
    pub fn calculate_output_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[LayerTypeCPU],
        cost_fn: CostFunction,
    ) {
        for i in 0..learn_data.node_values.len() {
            let dcost = cost_fn.call_d(
                &vec![learn_data.activation_values[i]],
                &vec![expected_outputs[i]],
            );
            let dactivation = self
                .activation_type
                .activate_derivative(learn_data.weighted_inputs[i]);
            learn_data.node_values[i] = dactivation * dcost;
        }
    }
    #[cfg(feature = "simd")]
    pub fn calculate_output_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[f32],
        cost_fn: CostFunction,
    ) {
        use wide::f32x8;

        const CHUNK_SIZE: usize = 8;

        let activation_vals = &learn_data.activation_values;
        let weighted_inputs = &learn_data.weighted_inputs;
        let node_vals = &mut learn_data.node_values;

        let chunks_activation = activation_vals.chunks_exact(CHUNK_SIZE);
        let chunks_weighted = weighted_inputs.chunks_exact(CHUNK_SIZE);
        let chunks_expected = expected_outputs.chunks_exact(CHUNK_SIZE);
        let mut chunks_node_vals = node_vals.chunks_exact_mut(CHUNK_SIZE);

        let remainder_activation = chunks_activation.remainder();
        let remainder_weighted = chunks_weighted.remainder();
        let remainder_expected = chunks_expected.remainder();

        for ((chunk_activation, chunk_weighted), (chunk_expected, chunk_node_vals)) in
            chunks_activation
                .zip(chunks_weighted)
                .zip(chunks_expected.zip(chunks_node_vals.by_ref()))
        {
            let act_vec = f32x8::from(chunk_activation);
            let weighted_vec = f32x8::from(chunk_weighted);
            let expected_vec = f32x8::from(chunk_expected);

            let dcost = cost_fn.call_simd_d(act_vec, expected_vec);
            let dactivation = self.activation_type.activate_derivative_simd(weighted_vec);

            let result = dactivation * dcost;

            let result_arr: [f32; CHUNK_SIZE] = result.into();
            chunk_node_vals.copy_from_slice(&result_arr);
        }

        let remainder_node_vals = chunks_node_vals.into_remainder();

        if !remainder_activation.is_empty() {
            for i in 0..remainder_activation.len() {
                let dcost =
                    cost_fn.call_d(&vec![remainder_activation[i]], &vec![remainder_expected[i]]);
                let dactivation = self
                    .activation_type
                    .activate_derivative(remainder_weighted[i]);
                remainder_node_vals[i] = dactivation * dcost;
            }
        }
    }

    pub fn calculate_hidden_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        prev_layer: &Layer,
        prev_node_cost_values: &[LayerTypeCPU],
    ) {
        learn_data.node_values.fill(0.0);

        for (prev_cost, weights_row) in prev_node_cost_values.iter().zip(prev_layer.weights.iter())
        {
            for (new_val, &weight) in learn_data.node_values.iter_mut().zip(weights_row.iter()) {
                *new_val += weight * prev_cost;
            }
        }
        for (new_val, &weighted_input) in learn_data
            .node_values
            .iter_mut()
            .zip(learn_data.weighted_inputs.iter())
        {
            *new_val *= self.activation_type.activate_derivative(weighted_input);
        }
    }
}

// All layers
pub fn forward(
    layers: &Vec<Layer>,
    inputs: &[LayerTypeCPU],
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) {
    pingpong.current.clear();
    pingpong.current.extend_from_slice(inputs);

    for layer in layers {
        pingpong.next.resize(layer.num_out_nodes, 0.0);
        layer.calculate_outputs(&pingpong.current, &mut pingpong.next);
        std::mem::swap(&mut pingpong.current, &mut pingpong.next);
    }

    // Simplicity, keep input at current, output at next
    std::mem::swap(&mut pingpong.current, &mut pingpong.next);

    if is_softmax_output {
        ActivationFunctionType::apply_softmax(&mut pingpong.next);
    }
}

pub fn calculate_cost(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    #[cfg(not(feature = "simd"))]
    {
        calculate_cost_scalar(layers, data, cost_fn, pingpong, is_softmax_output)
    }
    #[cfg(feature = "simd")]
    {
        calculate_cost_simd(layers, data, cost_fn, pingpong, is_softmax_output)
    }
}

#[cfg(not(feature = "simd"))]
fn calculate_cost_scalar(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    let total_cost: LayerTypeCPU = data
        .iter()
        .map(|dp| calculate_cost_datapoint(layers, dp, cost_fn, pingpong, is_softmax_output))
        .sum();

    let l2_penalty: LayerTypeCPU = layers
        .iter()
        .flat_map(|layer| layer.weights.iter())
        .flat_map(|matrix| matrix.iter())
        .map(|w| w.powi(2))
        .sum();

    (total_cost / (data.len() as LayerTypeCPU)) + (0.001 * l2_penalty)
}

#[cfg(not(feature = "simd"))]
fn calculate_cost_datapoint(
    layers: &Vec<Layer>,
    datapoint: &DataPoint,
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    forward(layers, &datapoint.inputs, pingpong, is_softmax_output);
    cost_fn.call(&pingpong.next, &datapoint.expected_outputs)
}

#[cfg(feature = "simd")]
fn calculate_cost_simd(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    let num_outputs = layers.last().unwrap().num_out_nodes;

    let total_cost: LayerTypeCPU = data
        .iter()
        .map(|datapoint| {
            forward(layers, &datapoint.inputs, pingpong, is_softmax_output);
            let mut sum = f32x8::splat(0.0);
            let mut i = 0;

            while i + 8 <= num_outputs {
                let pred = f32x8::from(&pingpong.next[i..i + 8]);
                let expected = f32x8::from(&datapoint.expected_outputs[i..i + 8]);
                sum += cost_fn.call_simd(pred, expected);
                i += 8;
            }

            let mut cost = sum.reduce_add();
            if i < num_outputs {
                cost += cost_fn.call(&pingpong.next[i..], &datapoint.expected_outputs[i..]);
            }
            cost
        })
        .sum();

    let l2_penalty: LayerTypeCPU = layers
        .iter()
        .flat_map(|layer| layer.weights.iter())
        .flat_map(|matrix| matrix.iter())
        .map(|w| w.powi(2))
        .sum();

    (total_cost / (data.len() as LayerTypeCPU)) + (0.001 * l2_penalty)
}
