use std::{fmt::Display, fs::File, io::Write, path::Path, sync::Arc, time::Duration};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::zneural_network::{
    datapoint::{DataPoint, TrainingData},
    neuralnetwork::{
        NNExpectedOutputs, NNExpectedOutputsRef, NNIsCorrectFn, NNOutputs, NNOutputsRef,
        NeuralNetwork,
    },
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum DatasetUsage {
    Training,
    Validation,
    Test,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct TrainingSession {
    pub nn: Option<NeuralNetwork>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub state: TrainingState,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learn_rate: f32,
    pub training_data: TrainingData,
}

impl Default for TrainingSession {
    fn default() -> Self {
        Self {
            nn: None,
            state: TrainingState::Idle,
            num_epochs: 2,
            batch_size: 1000,
            learn_rate: 0.2,
            training_data: TrainingData::default(),
        }
    }
}

impl TrainingSession {
    pub fn new(
        nn: Option<&NeuralNetwork>,
        training_data: TrainingData,
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
            training_data: training_data,
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

    pub fn set_training_data(&mut self, in_data: TrainingData) {
        self.training_data = in_data;
    }

    pub fn ready(&self) -> bool {
        self.nn.is_some()
            && self.training_data.get_in_out_dimensions().0 > 0
            && self.training_data.get_in_out_dimensions().1 > 0
            && self.num_epochs > 0
            && self.batch_size > 0
            && self.learn_rate > 0.0
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct TestResults {
    pub results: Vec<(DataPoint, NNOutputs)>, // results for each datapoint
    pub num_correct: i32,
    pub accuracy: Option<f32>,
    pub cost: f32,
}

impl TestResults {
    pub fn new(
        results: Vec<(DataPoint, NNOutputs)>,
        eval_correct_fn: Option<&NNIsCorrectFn>,
        avg_cost: f32,
    ) -> Self {
        let mut num_correct = 0;
        for (datapoint, outputs) in &results {
            let is_correct = NeuralNetwork::is_output_correct(
                outputs,
                &datapoint.expected_outputs,
                eval_correct_fn,
            );
            if is_correct {
                num_correct += 1;
            }
        }

        Self {
            num_correct: num_correct,
            accuracy: Some((num_correct as f32) / (results.len() as f32)),
            cost: avg_cost,
            results,
        }
    }

    pub fn save_results(&self, path: impl AsRef<Path>) -> Result<(), anyhow::Error> {
        let mut file = File::create(path.as_ref())?;

        let mut json = serde_json::to_string_pretty(&self.results)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl Display for TestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TestResults(\n\tnum_total: {}\n\tnum_correct: {}\n\taccuracy: {}\n\tcost: {}\n)",
            self.len(),
            self.num_correct,
            self.accuracy.unwrap_or_default(),
            self.cost
        )
    }
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

pub fn test_nn<'a>(
    nn: &'a mut NeuralNetwork,
    test_data: &[DataPoint],
) -> Result<&'a TestResults, anyhow::Error> {
    if test_data.len() >= 1
        && test_data.first().unwrap().inputs.len() == nn.graph_structure.input_nodes
        && test_data.first().unwrap().expected_outputs.len() == nn.graph_structure.output_nodes
    {
        log::info!("Start test_nn");
        let mut num_correct = 0;

        let mut results = Vec::with_capacity(test_data.len());
        for i in 0..test_data.len() {
            let mut datapoint = &test_data[i];
            let outputs = nn.calculate_outputs(&datapoint.inputs[..]);
            results.push((test_data[i].clone(), outputs));
        }

        let avg_cost = nn.calculate_costs(test_data);
        // let test_results = TestResults::new(results, None, avg_cost);
        let test_results = TestResults::new(
            results,
            Some(
                &|outputs: &NNOutputsRef, expected_outputs: &NNExpectedOutputsRef| {
                    const ESPILON: f32 = 0.001;

                    let mut is_correct = false;
                    for (output, expected_output) in outputs.iter().zip(expected_outputs) {
                        is_correct |= NeuralNetwork::is_normalized_within_tolerance(
                            *output,
                            *expected_output,
                            60.0,
                            Duration::from_secs(20 * 60),
                        );
                    }
                    is_correct
                },
            ),
            avg_cost,
        );
        nn.last_test_results = Some(test_results);
        Ok(&nn.last_test_results.as_ref().unwrap())
    } else {
        anyhow::bail!("Failed to test_nn")
    }
}
