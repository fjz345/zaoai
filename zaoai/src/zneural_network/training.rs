use std::{
    fmt::Display,
    fs::File,
    io::Write,
    path::Path,
    sync::mpsc::{Receiver, Sender},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use zaoai_types::ai_labels::LayerTypeCPU;

use crate::{
    app::NeuralNetworkType,
    zneural_network::{
        cpu::layer::calculate_cost,
        cpu::neuralnetwork_cpu::{NNOutputs, NeuralNetworkCPU, NeuralNetworkPingPong},
        datapoint::{DataPoint, TrainingData},
        is_correct::{ConfusionCategory, ConfusionEvaluator},
        thread::TrainingThreadPayload,
    },
};

use crate::gpu::neuralnetwork_gpu::NeuralNetworkGPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Display, PartialEq)]
pub enum FloatDecay {
    Exponential {
        rate: LayerTypeCPU,
    },
    StepDecay {
        step_size: usize,
        decay_factor: LayerTypeCPU, // 0.5 to halve every step_size
    },
    Linear {
        max_steps: usize,
        end_rate: LayerTypeCPU,
    },
    Cosine {
        max_steps: usize,
        min_val: LayerTypeCPU,
    },
}

impl Default for FloatDecay {
    fn default() -> Self {
        Self::Exponential { rate: 0.05 }
    }
}

impl FloatDecay {
    pub fn decay(&self, init_val: LayerTypeCPU, step: usize) -> LayerTypeCPU {
        match self {
            Self::Exponential { rate } => init_val * (-rate * step as LayerTypeCPU).exp(),
            Self::StepDecay {
                step_size,
                decay_factor,
            } => {
                let inv_decay_factor = 1.0 - decay_factor.clamp(0.0, 1.0);
                let exponent = (step / *step_size) as LayerTypeCPU;
                init_val * inv_decay_factor.powf(exponent)
            }
            Self::Linear {
                max_steps,
                end_rate,
            } => {
                let progress = step as LayerTypeCPU / *max_steps as LayerTypeCPU;
                if progress >= 1.0 {
                    *end_rate
                } else {
                    init_val * (1.0 - progress) + end_rate * progress
                }
            }
            Self::Cosine { max_steps, min_val } => {
                let progress = step as LayerTypeCPU / *max_steps as LayerTypeCPU;
                if progress >= 1.0 {
                    *min_val
                } else {
                    min_val
                        + 0.5
                            * (init_val - min_val)
                            * (1.0 + (std::f64::consts::PI as LayerTypeCPU * progress).cos())
                }
            }
        }
    }

    pub fn set_max_steps(&mut self, in_max_steps: usize) {
        match self {
            Self::Exponential { .. } | Self::StepDecay { .. } => {}
            Self::Linear { max_steps, .. } | Self::Cosine { max_steps, .. } => {
                *max_steps = in_max_steps
            }
        }
    }

    pub fn set_decay_rate(&mut self, rate: LayerTypeCPU) {
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
    pub cost: LayerTypeCPU,
    pub last_loss: LayerTypeCPU,
    pub num_merged: usize,
    pub dataset_usage: DatasetUsage,
    pub learn_rate: LayerTypeCPU,
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

#[allow(dead_code)]
impl AIResultMetadata {
    pub fn new(
        dataset_usage: DatasetUsage,
        cost: LayerTypeCPU,
        last_loss: LayerTypeCPU,
        learn_rate: LayerTypeCPU,
    ) -> Self {
        Self {
            cost,
            last_loss,
            num_merged: 1,
            dataset_usage,
            learn_rate,
            ..Default::default()
        }
    }

    pub fn from_correct(correct: usize, total: usize) -> Self {
        let correct = correct.min(total);
        let incorrect = total.saturating_sub(correct);

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

    pub fn from_accuracy(accuracy: f64, total_preds: usize) -> Self {
        assert!(accuracy <= 1.0, "Accuracy must be in the range 0.0..=1.0");
        assert!(accuracy >= 0.0, "Accuracy must be in the range 0.0..=1.0");
        if total_preds == 0 {
            return Self::from_correct(0, 0);
        }
        let accuracy = accuracy.clamp(0.0, 1.0);
        let correct = (accuracy * total_preds as f64).round() as usize;
        Self::from_correct(correct, total_preds)
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

        self.cost = (self.cost * (self.num_merged - 1) as LayerTypeCPU + other.cost)
            / self.num_merged as LayerTypeCPU;

        self
    }

    pub fn positive_instances(&self) -> usize {
        self.true_positives + self.false_negatives
    }

    pub fn negative_instances(&self) -> usize {
        self.true_negatives + self.false_positives
    }

    pub fn calc_accuracy(&self) -> f64 {
        let total = self.positive_instances() + self.negative_instances();
        if total == 0 {
            return 0.0;
        }
        let correct = self.true_positives + self.true_negatives;
        (correct as f64 / total as f64).clamp(0.0, 1.0)
    }

    pub fn calc_error_rate(&self) -> f64 {
        let total = self.positive_instances() + self.negative_instances();
        if total == 0 {
            return 0.0;
        }
        let incorrect = self.false_positives + self.false_negatives;
        (incorrect as f64 / total as f64).clamp(0.0, 1.0)
    }

    pub fn calc_true_positive_rate(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denominator as f64
    }

    pub fn calc_true_negative_rate(&self) -> f64 {
        let denominator = self.false_positives + self.false_negatives;
        if denominator == 0 {
            return 0.0;
        }
        self.true_negatives as f64 / denominator as f64
    }

    pub fn calc_positive_liklihood(&self) -> f64 {
        let false_positive_rate = 1.0 - self.calc_true_negative_rate();
        if false_positive_rate == 0.0 {
            return 0.0;
        }
        self.calc_true_positive_rate() / false_positive_rate
    }

    pub fn calc_negative_liklihood(&self) -> f64 {
        let true_negative_rate = self.calc_true_negative_rate();
        if true_negative_rate == 0.0 {
            return 0.0;
        }
        self.calc_true_positive_rate() / true_negative_rate
    }

    pub fn calc_f1_score(&self) -> f64 {
        let precision_denominator = self.true_positives + self.false_positives;
        let recall_denominator = self.true_positives + self.false_negatives;
        if precision_denominator == 0 || recall_denominator == 0 {
            return 0.0;
        }
        let precision = self.true_positives as f64 / precision_denominator as f64;
        let recall = self.true_positives as f64 / recall_denominator as f64;
        let denominator = precision + recall;
        if denominator == 0.0 {
            return 0.0;
        }
        2.0 * (precision * recall) / denominator
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct TrainingSession {
    #[cfg_attr(feature = "serde", serde(skip))]
    pub nn: Option<NeuralNetworkType>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub state: TrainingState,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learn_rate: LayerTypeCPU,
    pub learn_rate_decay: Option<FloatDecay>,
    pub learn_rate_decay_rate: LayerTypeCPU,
    pub training_data: TrainingData,
    pub is_correct_fn: ConfusionEvaluator,
    pub validation_each_epoch: usize,
}

impl TrainingSession {
    pub fn new(
        nn: Option<&NeuralNetworkType>,
        training_data: TrainingData,
        num_epochs: usize,
        batch_size: usize,
        learn_rate: LayerTypeCPU,
        learn_rate_decay: Option<FloatDecay>,
        learn_rate_decay_rate: LayerTypeCPU,
        is_correct_fn: ConfusionEvaluator,
        validation_each_epoch: usize,
    ) -> Self {
        Self {
            nn: nn.cloned(),
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

    pub fn set_nn(&mut self, nn: &NeuralNetworkType) {
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
    pub fn get_learn_rate(&self) -> LayerTypeCPU {
        self.learn_rate
    }
    pub fn set_training_data(&mut self, in_data: TrainingData) {
        self.training_data = in_data;
    }
    // pub fn ready(&self) -> bool {
    //     self.nn.is_some()
    //         && self.training_data.get_in_out_dimensions().0 > 0
    //         && self.training_data.get_in_out_dimensions().1 > 0
    //         && self.num_epochs > 0
    //         && self.batch_size > 0
    //         && self.learn_rate > 0.0
    // }
}

#[derive(Serialize)]
struct ResultNoInputs<'a> {
    expected_outputs: &'a [LayerTypeCPU],
    outputs: &'a [LayerTypeCPU],
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct TestResults {
    pub results: Vec<(DataPoint, NNOutputs<LayerTypeCPU>)>, // results for each datapoint
    pub num_correct: i32,
    pub accuracy: Option<LayerTypeCPU>,
    pub cost: LayerTypeCPU,
}

impl TestResults {
    pub fn new(
        results: Vec<(DataPoint, NNOutputs<LayerTypeCPU>)>,
        eval_correct_fn: ConfusionEvaluator,
        avg_cost: LayerTypeCPU,
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
            accuracy: Some((num_correct as LayerTypeCPU) / (results.len() as LayerTypeCPU)),
            cost: avg_cost,
            results,
        }
    }

    pub fn save_results(&self, path: impl AsRef<Path>) -> Result<(), anyhow::Error> {
        return self.save_results_no_inputs(path);

        // let mut file = File::create(path.as_ref())?;

        // let mut json = serde_json::to_string_pretty(&self.results)?;
        // file.write_all(json.as_bytes())?;

        // Ok(())
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
    // Abort,
}
