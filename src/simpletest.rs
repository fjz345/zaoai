use crate::datapoint::*;
use crate::layer::*;
use crate::neuralnetwork::*;
use crate::zneural_network::training::test_nn;
use crate::zneural_network::*;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub fn simple_test_nn(nn: &mut NeuralNetwork, training_data: &Vec<DataPoint>) {
    let num_datapoints: usize = 1000000;
    let mut dataset = create_2x2_test_datapoints(0, num_datapoints as i32);
    let mut training_data: Vec<DataPoint> = Vec::new();
    let mut validation_data: Vec<DataPoint> = Vec::new();
    let mut test_data: Vec<DataPoint> = Vec::new();
    split_datapoints(
        &dataset,
        [0.75, 0.9],
        &mut training_data,
        &mut validation_data,
        &mut test_data,
    );

    // nn.learn_slow(&training_data, 10, 1000, 1.0, Some(true));
    nn.learn(&training_data, 2, 10, 0.2, Some(false), None);

    let test_result = test_nn(nn, &training_data);

    nn.print();
}
