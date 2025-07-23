use serde::{Deserialize, Serialize};

use crate::zneural_network::{datapoint::DataPoint, neuralnetwork::NeuralNetwork};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum DatasetUsage {
    Training,
    Validation,
    Test,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AIResultMetadata {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub positive_instances: usize,
    pub negative_instances: usize,
    pub cost: f64,
    pub last_loss: f64,
    num_merged: usize,
    dataset_usage: DatasetUsage,
}

impl AIResultMetadata {
    pub fn new(dataset_usage: DatasetUsage, cost: f64, last_loss: f64) -> Self {
        Self {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
            positive_instances: 0,
            negative_instances: 0,
            cost: cost,
            last_loss: last_loss,
            num_merged: 1,
            dataset_usage,
        }
    }

    pub fn merge(&mut self, other: &AIResultMetadata) -> &mut Self {
        assert_eq!(
            self.dataset_usage, other.dataset_usage,
            "DatasetUsage must match"
        );

        self.num_merged += 1;
        self.true_positives += other.true_positives;
        self.true_negatives += other.true_negatives;
        self.false_positives += other.false_positives;
        self.false_negatives += other.false_negatives;
        self.positive_instances += other.positive_instances;
        self.negative_instances += other.negative_instances;
        self.last_loss = other.last_loss;
        self.cost =
            (self.cost * (self.num_merged - 1) as f64 + other.cost) / self.num_merged as f64;
        self
    }

    pub fn calc_accuracy(&self) -> f64 {
        (self.true_positives + self.true_negatives) as f64
            / (self.positive_instances + self.negative_instances) as f64
    }

    pub fn calc_error_rate(&self) -> f64 {
        (self.false_positives + self.false_negatives) as f64
            / (self.positive_instances + self.negative_instances) as f64
    }

    pub fn calc_true_positive_rate(&self) -> f64 {
        self.true_positives as f64 / (self.true_positives + self.false_positives) as f64
    }

    pub fn calc_true_negative_rate(&self) -> f64 {
        self.true_negatives as f64 / (self.false_positives + self.false_negatives) as f64
    }

    pub fn calc_positive_liklihood(&self) -> f64 {
        self.calc_true_positive_rate() as f64 / (1.0 - self.calc_true_negative_rate()) as f64
    }

    pub fn calc_negative_liklihood(&self) -> f64 {
        self.calc_true_positive_rate() as f64 / self.calc_true_negative_rate() as f64
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TrainingSession {
    pub nn: Option<NeuralNetwork>,
    #[serde(skip)]
    pub state: TrainingState,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learn_rate: f32,
    pub training_data: Vec<DataPoint>,
}

impl Default for TrainingSession {
    fn default() -> Self {
        Self {
            nn: None,
            state: TrainingState::Idle,
            num_epochs: 2,
            batch_size: 1000,
            learn_rate: 0.2,
            training_data: Vec::new(),
        }
    }
}

impl TrainingSession {
    pub fn new(
        nn: Option<&NeuralNetwork>,
        training_data: &[DataPoint],
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
    ) -> Self {
        let mut nn_option: Option<NeuralNetwork> = None;
        if nn.is_some() {
            nn_option = Some(nn.unwrap().clone());
        }

        Self {
            nn: nn_option,
            state: TrainingState::Idle,
            num_epochs: num_epochs,
            batch_size: batch_size,
            learn_rate: learn_rate,
            training_data: training_data.to_vec(),
        }
    }

    pub fn set_nn(&mut self, nn: &NeuralNetwork) {
        self.nn = Some(nn.clone());
    }

    pub fn set_state(&mut self, new_state: TrainingState) {
        self.state = new_state;
    }

    pub fn get_state(&self) -> TrainingState {
        self.state
    }

    pub fn get_num_epochs(&self) -> usize {
        self.num_epochs
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn get_learn_rate(&self) -> f32 {
        self.learn_rate
    }

    pub fn set_training_data(&mut self, in_data: &[DataPoint]) {
        self.training_data = in_data.to_vec();
    }

    pub fn ready(&self) -> bool {
        self.nn.is_some()
            && self.training_data.len() > 0
            && self.num_epochs > 0
            && self.batch_size > 0
            && self.learn_rate > 0.0
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct TestResults {
    pub num_datapoints: i32,
    pub num_correct: i32,
    pub accuracy: f32,
    pub cost: f32,
}

#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub enum TrainingState {
    #[default]
    Idle,
    StartTraining,
    Training,
    Finish,
    Abort,
}

// Returns a TestResult
pub fn test_nn(nn: &mut NeuralNetwork, test_data: &[DataPoint]) -> TestResults {
    if test_data.len() >= 1 {
        let mut num_correct = 0;

        for i in 0..test_data.len() {
            let mut datapoint = &test_data[i];
            let outputs = nn.calculate_outputs(&datapoint.inputs[..]);
            let result = NeuralNetwork::determine_output_result(&outputs);
            let result_expected =
                NeuralNetwork::determine_output_result(&datapoint.expected_outputs);

            let is_correct = result.0 == result_expected.0;
            if (is_correct) {
                num_correct += 1;
            }
        }

        let avg_cost = nn.calculate_cost(test_data);
        let test_result = TestResults {
            num_datapoints: test_data.len() as i32,
            num_correct: num_correct,
            accuracy: (num_correct as f32) / (test_data.len() as f32),
            cost: avg_cost,
        };

        nn.last_test_results = test_result;
        test_result
    } else {
        TestResults {
            num_datapoints: 0,
            num_correct: 0,
            accuracy: 0.0,
            cost: 0.0,
        }
    }
}
