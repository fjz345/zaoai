use rand::prelude::*;

fn sigmoid(in_value: f32) -> f32 {
    1.0 / (1.0 + (-in_value).exp())
}

fn relu(in_value: f32) -> f32 {
    in_value.max(0.0)
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
        for weight in weights.iter_mut() {
            weight.resize(num_in_nodes, 0.0);
        }
        let mut weights_cost_grads: Vec<Vec<f32>> = Vec::new();
        weights_cost_grads.reserve(num_out_nodes);
        for weights_cost_grad in weights_cost_grads.iter_mut() {
            weights_cost_grad.resize(num_in_nodes, 0.0);
        }

        // Initialize weights & biases
        for i in 0..(num_out_nodes - 1) {
            let rand_bias: f32 = random();
            biases[i] = rand_bias;

            for j in 0..(num_in_nodes - 1) {
                let rand_weight: f32 = random();
                weights[i][j] = rand_weight;
            }
        }

        Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
        }
    }

    fn activation_function(in_value: f32) -> f32 {
        relu(in_value)
        //sigmoid(in_value)
    }

    fn node_cost(output_activation: f32, expected_activation: f32) -> f32 {
        let error = output_activation - expected_activation;
        error * error
    }

    fn calculate_outputs(&self, activation_inputs: &[f32]) -> Vec<f32> {
        let mut activation_outputs: Vec<f32> = Vec::new();
        activation_outputs.resize(self.num_out_nodes, 0.0);

        for (output_node, output) in activation_outputs.iter_mut().enumerate() {
            // First apply bias for the output node
            let mut weighted_input = self.biases[output_node];

            // Then apply input node weights
            for j in 0..self.num_in_nodes {
                weighted_input += activation_inputs[j] * self.weights[output_node][j];
            }

            *output = weighted_input;
        }

        activation_outputs
    }

    fn apply_cost_gradient(&mut self, learn_rate: f32) {
        for node in 0..self.num_out_nodes {
            self.biases[node] -= self.biases_cost_grads[node] * learn_rate;

            for weight in 0..self.num_in_nodes {
                self.weights[node][weight] -= self.weights_cost_grads[node][weight] * learn_rate;
            }
        }
    }
}

pub struct GraphStructure {
    input_nodes: usize,
    hidden_layers: Vec<usize>, // contais nodes
    output_nodes: usize,
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
            hidden_layers: hidden_nodes,
            output_nodes,
        }
    }

    fn validate(&self) -> bool {
        let mut is_valid = true;

        if self.input_nodes < 1 {
            is_valid = false;
        } else if self.output_nodes < 1 {
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

#[derive(Copy, Clone)]
pub struct DataPoint {
    inputs: [f32; 2],
    expected_outputs: [f32; 2],
}

pub struct NeuralNetwork {
    graph_structure: GraphStructure,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(graph_structure: GraphStructure) -> NeuralNetwork {
        let mut layers: Vec<Layer> = Vec::new();
        let mut prev_out_size = graph_structure.input_nodes;

        // Input nodes are not layers in the neural network.

        // Create Hidden layers
        for i in &graph_structure.hidden_layers[..] {
            layers.push(Layer::new(prev_out_size, *i));
        }

        // Create Output layer
        layers.push(Layer::new(prev_out_size, graph_structure.output_nodes));

        NeuralNetwork {
            graph_structure,
            layers,
        }
    }

    pub fn learn(&mut self, training_data: &[DataPoint], learn_rate: f32) {
        let h: f32 = 0.0001;

        let original_cost = NeuralNetwork::calculate_cost(&self.layers, training_data);

        let mut calc_cost_layers = self.layers.to_vec();

        // Calculate cost gradients for layers
        for layer in self.layers.iter_mut() {
            for node in 0..layer.num_out_nodes {
                // Biases
                layer.biases[node] += h;
                let dcost =
                    NeuralNetwork::calculate_cost(&calc_cost_layers, training_data) - original_cost;
                layer.biases[node] -= h;
                layer.biases_cost_grads[node] = dcost / h;

                // Weightsimage.png
                for weight in 0..layer.num_in_nodes {
                    layer.weights[node][weight] += h;
                    let dcost = NeuralNetwork::calculate_cost(&calc_cost_layers, training_data)
                        - original_cost;
                    layer.weights[node][weight] -= h;
                    layer.weights_cost_grads[node][weight] = dcost / h;
                }
            }
        }

        // Adjust weights & biases
        self.apply_cost_gradients(learn_rate);
    }

    fn apply_cost_gradients(&mut self, learn_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.apply_cost_gradient(learn_rate);
        }
    }

    pub fn calculate_outputs(layers: &[Layer], inputs: &mut &[f32]) {
        for layer in layers.iter() {
            layer.calculate_outputs(inputs);
        }
    }

    fn calculate_cost_data_point(layers: &[Layer], data_point: DataPoint) -> f32 {
        let mut cost: f32 = 0.0;

        // Calculate the output of the neural network
        let mut data_point_output = data_point.inputs.clone();
        NeuralNetwork::calculate_outputs(layers, &mut &data_point_output[..]);

        // Calculate cost by comparing difference between output and expected output
        let output_layer = layers.last().unwrap();
        for output_node in 0..output_layer.num_out_nodes {
            cost += Layer::node_cost(
                data_point_output[output_node],
                data_point.expected_outputs[output_node],
            );
        }

        cost
    }

    pub fn calculate_cost(layers: &[Layer], data: &[DataPoint]) -> f32 {
        if data.len() <= 0 {
            panic!("Input data was len: {}", data.len());
        }

        let mut cost: f32 = 0.0;

        for data_point in &data[..] {
            let copy: DataPoint = data_point.clone();
            cost += NeuralNetwork::calculate_cost_data_point(layers, copy);
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
        println!("----------------------------------");
    }
}
