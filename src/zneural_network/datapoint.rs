use std::path::PathBuf;

use crate::{
    neuralnetwork::*,
    sound::{decode_samples_from_file, S_SPECTOGRAM_NUM_BINS},
};

use rand::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sonogram::{SpecOptionsBuilder, Spectrogram};
use zaoai_types::ai_labels::ZaoaiLabel;

pub struct AnimeDataPoint {
    pub path: PathBuf,
    pub spectogram: Spectrogram,
    pub expected_outputs: Vec<f32>,
}

impl AnimeDataPoint {
    pub fn into_data_point(self, img_width: usize, img_height: usize) -> DataPoint {
        let buffer =
            self.spectogram
                .to_buffer(sonogram::FrequencyScale::Log, img_width, img_height);
        DataPoint {
            inputs: buffer,
            expected_outputs: self.expected_outputs,
        }
    }
}

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

pub fn create_test_spectogram(path: &PathBuf) -> Vec<AnimeDataPoint> {
    let (samples, sample_rate) = decode_samples_from_file(&path.as_path());

    let mut spectrobuilder = SpecOptionsBuilder::new(S_SPECTOGRAM_NUM_BINS)
        .load_data_from_memory_f32(samples, sample_rate)
        .build()
        .unwrap();
    let mut spectogram = spectrobuilder.compute();

    let new_point = AnimeDataPoint {
        path: path.to_path_buf(),
        spectogram,
        expected_outputs: vec![0.08936, 0.1510],
    };

    vec![new_point]
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

pub enum TrainingData {
    Physical(TrainingDataset),
    Virtual(VirtualTrainingDataset), // indirect Training data
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct VirtualTrainingDataset {
    pub virtual_dataset: Vec<ZaoaiLabel>,
    pub thresholds: [f64; 2],
}

pub struct VirtualTrainingBatchIter<'a> {
    dataset: &'a VirtualTrainingDataset,
    batch_size: usize,
    index: usize,
    end: usize,
}

impl<'a> Iterator for VirtualTrainingBatchIter<'a> {
    type Item = Vec<DataPoint>;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.dataset.virtual_dataset.len().min(self.end);

        if self.index >= len {
            return None;
        }

        let batch_end = (self.index + self.batch_size).min(len);
        let slice = &self.dataset.virtual_dataset[self.index..batch_end];

        // Convert the slice of ZaoaiLabel to Vec<DataPoint>
        let batch: Vec<DataPoint> = slice
            .iter()
            .map(|label| zaoai_label_to_datapoint(label))
            .collect();

        self.index = batch_end;

        Some(batch)
    }
}

fn zaoai_label_to_datapoint(label: &ZaoaiLabel) -> DataPoint {
    DataPoint {
        inputs: vec![],
        expected_outputs: vec![],
    }
}

impl VirtualTrainingDataset {
    pub fn training_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.virtual_dataset.len() as f64)) as usize;

        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: 0,
            end: traning_data_end,
        }
    }

    pub fn validation_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.virtual_dataset.len() as f64)) as usize;
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.virtual_dataset.len() as f64)) as usize;

        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: validadtion_data_end,
            end: traning_data_end,
        }
    }

    pub fn test_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.virtual_dataset.len() as f64)) as usize;
        let test_data_end: usize = self.virtual_dataset.len();

        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: validadtion_data_end,
            end: test_data_end,
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct TrainingDataset {
    pub full_dataset: Vec<DataPoint>,
    pub thresholds: [f64; 2],
}

impl TrainingDataset {
    pub fn new(datapoints: &[DataPoint]) -> Self {
        Self {
            full_dataset: datapoints.to_vec(),
            thresholds: [0.6, 0.8],
        }
    }

    pub fn new_from_splits(
        training_split: &[DataPoint],
        validation_split: &[DataPoint],
        test_split: &[DataPoint],
    ) -> Self {
        let training_split_len = training_split.len();
        let validation_split_len = validation_split.len();
        let test_split_len = test_split.len();

        let full_dataset: Vec<DataPoint> = training_split
            .iter()
            .chain(validation_split.iter())
            .chain(test_split.iter())
            .cloned()
            .collect();
        let thresholds = [
            ((training_split_len as f64) / full_dataset.len() as f64),
            (((training_split_len + validation_split_len) as f64) / full_dataset.len() as f64),
        ];

        let new_self = Self {
            full_dataset: full_dataset,
            thresholds: thresholds,
        };

        assert_eq!(new_self.training_split(), training_split);
        assert_eq!(new_self.validation_split(), validation_split);
        assert_eq!(new_self.test_split(), test_split);
        new_self
    }

    pub fn training_split(&self) -> &[DataPoint] {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.full_dataset.len() as f64)) as usize;

        &self.full_dataset[0..traning_data_end]
    }

    pub fn validation_split(&self) -> &[DataPoint] {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.full_dataset.len() as f64)) as usize;
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.full_dataset.len() as f64)) as usize;

        &self.full_dataset[traning_data_end..validadtion_data_end]
    }

    pub fn test_split(&self) -> &[DataPoint] {
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.full_dataset.len() as f64)) as usize;
        let test_data_end: usize = self.full_dataset.len();

        &self.full_dataset[validadtion_data_end..test_data_end]
    }

    pub fn iter(&self) -> impl Iterator<Item = &DataPoint> + '_ {
        self.training_split()
            .iter()
            .chain(self.validation_split().iter())
            .chain(self.test_split().iter())
    }

    pub fn get_thresholds(&self) -> [f64; 2] {
        self.thresholds
    }

    // Returns the number of (in, out) nodes needed in layers
    pub fn get_dimensions(&self) -> (usize, usize) {
        if self.full_dataset.len() >= 1 {
            (
                self.full_dataset[0].inputs.len(),
                self.full_dataset[0].expected_outputs.len(),
            )
        } else {
            (0, 0)
        }
    }
}
