use std::{
    fmt::Display,
    fs::File,
    io::Write,
    path::Path,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Duration,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;

use crate::zneural_network::{
    datapoint::{DataPoint, TrainingData},
    is_correct::ConfusionEvaluator,
    neuralnetwork::{
        NNExpectedOutputs, NNExpectedOutputsRef, NNOutputs, NNOutputsRef, NeuralNetwork,
    },
    thread::TrainingThreadPayload,
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Default)]
pub enum DatasetUsage {
    #[default]
    NotSet,
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
    pub cost: f64,
    pub last_loss: f64,
    pub num_merged: usize,
    pub dataset_usage: DatasetUsage,
    pub learn_rate: f32,
}

impl Default for AIResultMetadata {
    fn default() -> Self {
        Self {
            true_positives: Default::default(),
            true_negatives: Default::default(),
            false_positives: Default::default(),
            false_negatives: Default::default(),
            cost: Default::default(),
            last_loss: Default::default(),
            num_merged: 1,
            dataset_usage: Default::default(),
            learn_rate: Default::default(),
        }
    }
}

impl AIResultMetadata {
    pub fn new(dataset_usage: DatasetUsage, cost: f64, last_loss: f64, learn_rate: f32) -> Self {
        Self {
            cost: cost,
            last_loss: last_loss,
            num_merged: 1,
            dataset_usage,
            learn_rate,
            ..Default::default()
        }
    }

    pub fn from_accuracy(accuracy: f64, total_preds: usize) -> Self {
        let correct = (accuracy * total_preds as f64).round() as usize;
        let incorrect = total_preds - correct;

        // We'll fake these with symmetry
        let true_positives = correct / 2;
        let true_negatives = correct - true_positives;
        let false_positives = incorrect / 2;
        let false_negatives = incorrect - false_positives;

        Self {
            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
            num_merged: 1,
            dataset_usage: DatasetUsage::Test,
            ..Default::default()
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
        self.last_loss = other.last_loss;
        self.learn_rate = other.learn_rate;
        self.cost =
            (self.cost * (self.num_merged - 1) as f64 + other.cost) / self.num_merged as f64;
        self
    }

    pub fn add_counts(&mut self, tp: usize, tn: usize, fp: usize, fn_: usize) {
        self.true_positives += tp;
        self.true_negatives += tn;
        self.false_positives += fp;
        self.false_negatives += fn_;
    }

    pub fn positive_instances(&self) -> usize {
        self.true_positives + self.false_negatives
    }

    pub fn negative_instances(&self) -> usize {
        self.true_negatives + self.false_positives
    }

    pub fn calc_accuracy(&self) -> f64 {
        (self.true_positives + self.true_negatives) as f64
            / (self.positive_instances() + self.negative_instances()) as f64
    }

    pub fn calc_error_rate(&self) -> f64 {
        (self.false_positives + self.false_negatives) as f64
            / (self.positive_instances() + self.negative_instances()) as f64
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

    pub fn calc_f1_score(&self) -> f64 {
        let precision =
            self.true_positives as f64 / (self.true_positives + self.false_positives) as f64;
        let recall =
            self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64;
        2.0 * (precision * recall) / (precision + recall)
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
    pub learn_rate_decay: Option<FloatDecay>,
    pub learn_rate_decay_rate: f32,
    pub training_data: TrainingData,
    pub is_correct_fn: ConfusionEvaluator,
    pub validation_each_epoch: usize,
}

impl TrainingSession {
    pub fn new(
        nn: Option<&NeuralNetwork>,
        training_data: TrainingData,
        num_epochs: usize,
        batch_size: usize,
        learn_rate: f32,
        learn_rate_decay: Option<FloatDecay>,
        learn_rate_decay_rate: f32,
        is_correct_fn: ConfusionEvaluator,
        validation_each_epoch: usize,
    ) -> Self {
        let mut nn_option: Option<NeuralNetwork> = None;
        if nn.is_some() {
            nn_option = Some(nn.unwrap().clone());
        }

        Self {
            nn: nn_option,
            state: TrainingState::Idle,
            num_epochs,
            batch_size,
            learn_rate,
            training_data,
            learn_rate_decay,
            learn_rate_decay_rate,
            is_correct_fn,
            validation_each_epoch,
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

#[derive(Serialize)]
struct ResultNoInputs<'a> {
    expected_outputs: &'a [f32],
    outputs: &'a [f32],
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
        eval_correct_fn: ConfusionEvaluator,
        avg_cost: f32,
    ) -> Self {
        let mut num_correct = 0;
        for (datapoint, outputs) in &results {
            let confusion_category = eval_correct_fn.evaluate(outputs, &datapoint.expected_outputs);
            match confusion_category {
                super::is_correct::ConfusionCategory::TruePositive => num_correct += 1,
                super::is_correct::ConfusionCategory::TrueNegative => num_correct += 1,
                super::is_correct::ConfusionCategory::FalsePositive => {}
                super::is_correct::ConfusionCategory::FalseNegative => {}
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
        return self.save_results_no_inputs(path);

        let mut file = File::create(path.as_ref())?;

        let mut json = serde_json::to_string_pretty(&self.results)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    pub fn save_results_no_inputs(&self, path: impl AsRef<Path>) -> Result<(), anyhow::Error> {
        let mut file = File::create(path.as_ref())?;

        // Strip out inputs from each result
        let stripped_results: Vec<ResultNoInputs> = self
            .results
            .iter()
            .map(|(datapoint, outputs)| ResultNoInputs {
                expected_outputs: &datapoint.expected_outputs,
                outputs,
            })
            .collect();

        let json = serde_json::to_string_pretty(&stripped_results)?;
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
    is_correct_fn: ConfusionEvaluator,
    tx_test_metadata: Option<Sender<TrainingThreadPayload>>,
    tx_abort: Option<Receiver<()>>,
) -> Result<&'a TestResults, anyhow::Error> {
    if test_data.len() >= 1
        && test_data.first().unwrap().inputs.len() == nn.graph_structure.input_nodes
        && test_data.first().unwrap().expected_outputs.len() == nn.graph_structure.output_nodes
    {
        log::info!("Start test_nn");

        let mut metadata = AIResultMetadata::new(DatasetUsage::Test, 0.0, 0.0, 0.0);

        let mut results = Vec::with_capacity(test_data.len());
        for i in 0..test_data.len() {
            let mut datapoint = &test_data[i];
            let outputs = nn.calculate_outputs(&datapoint.inputs[..]);

            if let Some(tx_test_metadata) = &tx_test_metadata {
                let cost = nn.calculate_costs(std::slice::from_ref(&test_data[i]));
                let mut metadata_point =
                    AIResultMetadata::new(DatasetUsage::Test, cost as f64, cost as f64, 0.0);

                let confusion = is_correct_fn.evaluate(&outputs, &datapoint.expected_outputs);
                match confusion {
                    crate::zneural_network::is_correct::ConfusionCategory::TruePositive => {
                        metadata_point.true_positives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::TrueNegative => {
                        metadata_point.true_negatives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::FalsePositive => {
                        metadata_point.false_positives += 1
                    }
                    crate::zneural_network::is_correct::ConfusionCategory::FalseNegative => {
                        metadata_point.false_negatives += 1
                    }
                }

                metadata.merge(&metadata_point);

                tx_test_metadata
                    .send(TrainingThreadPayload {
                        payload_index: i,
                        payload_max_index: test_data.len(),
                        training_metadata: metadata.clone(),
                    })
                    .unwrap();

                if let Some(abort) = &tx_abort {
                    if abort.try_recv().is_ok() {
                        log::info!("Abort Recieved, stopping test_nn...");
                        anyhow::bail!("Aborted")
                    }
                }
            }

            results.push((test_data[i].clone(), outputs));
        }

        let cost = nn.calculate_costs(test_data);
        // let test_results = TestResults::new(results, None, avg_cost);
        let test_results = TestResults::new(results, is_correct_fn, cost);
        nn.last_test_results = Some(test_results);
        Ok(&nn.last_test_results.as_ref().unwrap())
    } else {
        anyhow::bail!("Failed to test_nn")
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Display, PartialEq)]
pub enum FloatDecay {
    Exponential {
        rate: f32, // decay_rate is now embedded
    },
    StepDecay {
        step_size: usize,
        decay_factor: f32, // e.g. 0.5 to halve every step_size
    },
    Linear {
        max_steps: usize,
        end_rate: f32,
    },
    Cosine {
        max_steps: usize,
        min_val: f32,
    },
}

impl Default for FloatDecay {
    fn default() -> Self {
        Self::Exponential { rate: 0.05 }
    }
}

impl FloatDecay {
    pub fn decay(&self, init_val: f32, step: usize) -> f32 {
        match self {
            Self::Exponential { rate } => init_val * (-rate * step as f32).exp(),
            Self::StepDecay {
                step_size,
                decay_factor,
            } => {
                let exponent = (step / *step_size) as f32;
                init_val * decay_factor.powf(exponent)
            }
            Self::Linear {
                max_steps,
                end_rate,
            } => {
                let progress = step as f32 / *max_steps as f32;
                if progress >= 1.0 {
                    *end_rate
                } else {
                    // Linearly interpolate between init_val and end_rate
                    init_val * (1.0 - progress) + end_rate * progress
                }
            }
            Self::Cosine { max_steps, min_val } => {
                let progress = step as f32 / *max_steps as f32;
                if progress >= 1.0 {
                    *min_val
                } else {
                    min_val
                        + 0.5
                            * (init_val - min_val)
                            * (1.0 + (std::f32::consts::PI * progress).cos())
                }
            }
        }
    }

    pub fn set_decay_rate(&mut self, rate: f32) {
        match self {
            Self::Exponential { rate: r } => *r = rate,
            Self::StepDecay { decay_factor, .. } => *decay_factor = rate,
            Self::Linear { end_rate, .. } => *end_rate = rate,
            Self::Cosine { min_val, .. } => *min_val = rate,
        }
    }

    pub fn uses_decay_rate(&self) -> bool {
        matches!(
            self,
            Self::Exponential { .. }
                | Self::StepDecay { .. }
                | Self::Linear { .. }
                | Self::Cosine { .. }
        )
    }
}
