use crate::neuralnetwork::*;

use rand::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataPoint {
    pub inputs: Vec<f32>,
    pub expected_outputs: Vec<f32>,
}

fn calculate_y_for_datapoint(x1: f32, x2: f32) -> (f32, f32) {
    let y1: f32;
    let y2: f32;

    // For simplicity assume that x2_max == x1_max
    let f_x1x2 = 1.0 - x1;

    // y = 0 under f, 1 over f.
    if x2 < f_x1x2 {
        y1 = 0.0;
        y2 = 1.0;
    } else {
        y1 = 1.0;
        y2 = 0.0;
    }

    (y1, y2)
}

pub fn create_2x2_test_datapoints(seed: u64, num_datapoints: i32) -> Vec<DataPoint> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1_min = 0.0;
    let x1_max = 1.0;

    let x2_min = 0.0;
    let x2_max = 1.0;

    let mut result: Vec<DataPoint> = Vec::new();
    for i in 0..num_datapoints {
        // Generate x1 & x2
        let x1_rand = rng.gen_range(x1_min..x1_max);
        let x2_rand = rng.gen_range(x2_min..x2_max);

        // Calculate y1 & y2
        let (y1, y2) = calculate_y_for_datapoint(x1_rand, x2_rand);

        result.push(DataPoint {
            inputs: vec![x1_rand, x2_rand],
            expected_outputs: vec![y1, y2],
        });
    }

    result
}

pub fn split_datapoints(
    datapoints: &[DataPoint],
    thresholds: [f64; 2],
    out_training_datapoints: &mut Vec<DataPoint>,
    out_validation_datapoints: &mut Vec<DataPoint>,
    out_test_datapoints: &mut Vec<DataPoint>,
) {
    /* Layout
         threshold[0]     ...[1]
    --------------------------------
    | Training | Validation | Test |
    --------------------------------
    */

    assert!(0.0 <= thresholds[0], "Invalid Threshold");
    assert!(thresholds[0] <= thresholds[1], "Invalid Threshold");
    assert!(thresholds[1] <= 1.0, "Invalid Threshold");

    let traning_data_end: usize = (thresholds[0] * (datapoints.len() as f64)).floor() as usize;
    let validadtion_data_end: usize = (thresholds[1] * (datapoints.len() as f64)).floor() as usize;
    let test_data_end: usize = datapoints.len();

    datapoints[0..traning_data_end].clone_into(out_training_datapoints);
    datapoints[traning_data_end..validadtion_data_end].clone_into(out_validation_datapoints);
    datapoints[validadtion_data_end..test_data_end].clone_into(out_test_datapoints);
}
