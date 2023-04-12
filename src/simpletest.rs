use crate::zneural_network::*;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn CalculateYForDataPoint(x1: f32, x2: f32) -> (f32, f32) {
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

pub fn CreateDataPoints(seed: u64, num_datapoints: i32) -> Vec<DataPoint> {
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
        let (y1, y2) = CalculateYForDataPoint(x1_rand, x2_rand);

        result.push(DataPoint {
            inputs: [x1_rand, x2_rand],
            expected_outputs: [y1, y2],
        });
    }

    result
}

pub fn TestNNOld(nn: &mut NeuralNetwork) {
    let num_datapoints: usize = 1000000;
    let mut datapoints = CreateDataPoints(0, num_datapoints as i32);
    //println!("DataPoints:\n {:#?}", datapoints);

    // use 50/50 as traning data10
    nn.learn_old(&datapoints[0..num_datapoints / 2], 1000, 0.1, Some(true));

    let test_result = nn.test(&datapoints[num_datapoints / 2..num_datapoints]);

    nn.print();
}

pub fn TestNN(nn: &mut NeuralNetwork) {
    let num_datapoints: usize = 1000000;
    let mut datapoints = CreateDataPoints(0, num_datapoints as i32);
    //println!("DataPoints:\n {:#?}", datapoints);

    // use 50/50 as traning data10
    nn.learn(&datapoints[0..num_datapoints / 2], 1000, 0.1, Some(true));

    let test_result = nn.test(&datapoints[num_datapoints / 2..num_datapoints]);

    nn.print();
}
