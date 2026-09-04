use zaoai_types::ai_labels::LayerTypeCPU;

use crate::zneural_network::{
    activation::ActivationFunctionType,
    cpu::layer::{Layer, LayerLearnData},
};
#[cfg(not(feature = "simd"))]
use crate::zneural_network::{
    cost::CostFunction,
    cpu::{layer::forward, neuralnetwork_cpu::NeuralNetworkPingPong},
    datapoint::DataPoint,
};

impl Layer {
    pub fn apply_activation_scalar(
        weighted_inputs: &mut [LayerTypeCPU],
        t: ActivationFunctionType,
    ) {
        weighted_inputs
            .iter_mut()
            .for_each(|x| *x = t.activate_scalar(*x));
    }

    pub fn fill_learn_data_scalar(
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
            *act = self.activation_type.activate_scalar(*w_in);
        }
    }

    pub fn compute_weighted_inputs_scalar(
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

    pub fn calculate_outputs_scalar(&self, inputs: &[LayerTypeCPU], outputs: &mut [LayerTypeCPU]) {
        self.compute_weighted_inputs_scalar(inputs, outputs);
        Self::apply_activation(outputs, self.activation_type);
    }

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
                .activate_derivative_scalar(learn_data.weighted_inputs[i]);
            learn_data.node_values[i] = dactivation * dcost;
        }
    }

    fn update_cost_gradient_for_node_scalar(
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
                    Self::update_cost_gradient_for_node_scalar(
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
                Self::update_cost_gradient_for_node_scalar(
                    weight_grad_row,
                    bias_grad,
                    node_value,
                    inputs,
                    num_in_nodes,
                );
            }
        }
    }
}

#[cfg(not(feature = "simd"))]
pub fn calculate_cost_scalar(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    let total_cost: LayerTypeCPU = data
        .iter()
        .map(|dp| calculate_cost_datapoint_scalar(layers, dp, cost_fn, pingpong, is_softmax_output))
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
pub fn calculate_cost_datapoint_scalar(
    layers: &Vec<Layer>,
    datapoint: &DataPoint,
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    forward(layers, &datapoint.inputs, pingpong, is_softmax_output);
    cost_fn.call(&pingpong.next, &datapoint.expected_outputs)
}
