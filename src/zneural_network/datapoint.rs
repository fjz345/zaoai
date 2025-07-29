use std::{
    fs::File,
    path::{Path, PathBuf},
    usize,
};

use crate::neuralnetwork::*;

use anyhow::{Context, Result};
use rand::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use zaoai_types::{
    ai_labels::{AnimeDataPoint, ZaoaiLabel},
    file::relative_after,
    spectrogram::generate_spectogram,
    FrequencyScale,
};
use zaoai_types::{
    sound::{decode_samples_from_file, S_SPECTOGRAM_NUM_BINS},
    spectrogram::load_spectrogram,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataPoint {
    pub inputs: Vec<f32>,
    pub expected_outputs: Vec<f32>,
}

impl DataPoint {
    pub fn from_anime_data_point(
        anime_data_point: AnimeDataPoint,
        img_width: usize,
        img_height: usize,
    ) -> DataPoint {
        let buffer =
            anime_data_point
                .spectogram
                .to_buffer(FrequencyScale::Log, img_width, img_height);
        DataPoint {
            inputs: buffer,
            expected_outputs: anime_data_point.expected_outputs,
        }
    }
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TrainingData {
    Physical(TrainingDataset),
    Virtual(VirtualTrainingDataset), // indirect Training data
}

impl TrainingData {
    const SPECTOGRAM_WIDTH: usize = 512;
    const SPECTOGRAM_HEIGHT: usize = 512;

    pub fn get_thresholds(&self) -> [f64; 2] {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.thresholds,
            TrainingData::Virtual(virtual_training_dataset) => virtual_training_dataset.thresholds,
        }
    }
    pub fn set_thresholds(&mut self, t0: f64, t1: f64) {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.thresholds = [t0, t1],
            TrainingData::Virtual(virtual_training_dataset) => {
                virtual_training_dataset.thresholds = [t0, t1]
            }
        }
    }
    pub fn get_dimensions(&self) -> (usize, usize) {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.get_dimensions(),
            TrainingData::Virtual(virtual_training_dataset) => {
                virtual_training_dataset.get_dimensions()
            }
        }
    }

    pub fn len(&self) -> usize {
        let len = match self {
            TrainingData::Physical(training_dataset) => training_dataset.full_dataset.len(),
            TrainingData::Virtual(virtual_training_dataset) => {
                virtual_training_dataset.virtual_dataset.len()
            }
        };

        assert_eq!(
            len,
            self.training_split_len() + self.validation_split_len() + self.test_split_len()
        );
        len
    }

    // If out of memory is an issue, need to change return to delay zaoai_label_to_datapoint before learn_batch
    pub fn training_split(&self) -> Vec<DataPoint> {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.training_split().to_vec(),
            TrainingData::Virtual(virtual_training_dataset) => {
                let (start, end) = virtual_training_dataset.get_training_start_end();
                virtual_training_dataset.virtual_dataset[start..end]
                    .to_vec()
                    .iter()
                    .map(|f| {
                        zaoai_label_to_datapoint(
                            virtual_training_dataset.path.clone(),
                            f,
                            [Self::SPECTOGRAM_WIDTH, Self::SPECTOGRAM_HEIGHT],
                        )
                        .with_context(|| {
                            format!("failed to zaoai_label_to_datapoint \nDatapoint\n{:?}", f)
                        })
                        .unwrap()
                    })
                    .collect()
            }
        }
    }

    pub fn validation_split(&self) -> Vec<DataPoint> {
        match self {
            TrainingData::Physical(training_dataset) => {
                training_dataset.validation_split().to_vec()
            }
            TrainingData::Virtual(virtual_training_dataset) => {
                let (start, end) = virtual_training_dataset.get_validation_start_end();
                virtual_training_dataset.virtual_dataset[start..end]
                    .to_vec()
                    .iter()
                    .map(|f| {
                        zaoai_label_to_datapoint(
                            virtual_training_dataset.path.clone(),
                            f,
                            [Self::SPECTOGRAM_WIDTH, Self::SPECTOGRAM_HEIGHT],
                        )
                        .expect("failed to zaoai_label_to_datapoint")
                    })
                    .collect()
            }
        }
    }

    pub fn test_split(&self) -> Vec<DataPoint> {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.test_split().to_vec(),
            TrainingData::Virtual(virtual_training_dataset) => {
                let (start, end) = virtual_training_dataset.get_test_start_end();
                virtual_training_dataset.virtual_dataset[start..end]
                    .to_vec()
                    .iter()
                    .map(|f| {
                        zaoai_label_to_datapoint(
                            virtual_training_dataset.path.clone(),
                            f,
                            [Self::SPECTOGRAM_WIDTH, Self::SPECTOGRAM_HEIGHT],
                        )
                        .expect("failed to zaoai_label_to_datapoint")
                    })
                    .collect()
            }
        }
    }

    pub fn training_split_len(&self) -> usize {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.training_split().len(),
            TrainingData::Virtual(virtual_training_dataset) => {
                virtual_training_dataset.training_len()
            }
        }
    }
    pub fn validation_split_len(&self) -> usize {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.validation_split().len(),
            TrainingData::Virtual(virtual_training_dataset) => {
                virtual_training_dataset.validation_len()
            }
        }
    }
    pub fn test_split_len(&self) -> usize {
        match self {
            TrainingData::Physical(training_dataset) => training_dataset.test_split().len(),
            TrainingData::Virtual(virtual_training_dataset) => virtual_training_dataset.test_len(),
        }
    }
}

impl Default for TrainingData {
    fn default() -> Self {
        TrainingData::Physical(TrainingDataset::default())
    }
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct VirtualTrainingDataset {
    pub path: PathBuf,
    pub virtual_dataset: Vec<ZaoaiLabel>,
    pub thresholds: [f64; 2],
}

#[derive(Debug)]
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
        if self.index == 2255 && batch_end == 2254 {
            dbg!(&self);
        }
        let slice = &self.dataset.virtual_dataset[self.index..batch_end];

        const SPECTOGRAM_WIDTH: usize = 512;
        const SPECTOGRAM_HEIGHT: usize = 512;

        // Convert the slice of ZaoaiLabel to Vec<DataPoint>
        let batch: Vec<DataPoint> = slice
            .iter()
            .map(|label| {
                zaoai_label_to_datapoint(
                    self.dataset.path.clone(),
                    label,
                    [SPECTOGRAM_WIDTH, SPECTOGRAM_HEIGHT],
                )
                .expect("failed to zaoai_label_to_datapoint")
            })
            .collect();

        self.index = batch_end;

        Some(batch)
    }
}

// Probably should multithread so speed this up...
fn zaoai_label_to_datapoint(
    source_path: impl AsRef<Path>,
    label: &ZaoaiLabel,
    spectogram_dim: [usize; 2],
) -> Result<DataPoint> {
    let relative_path = relative_after(&label.path, &label.path_source).unwrap();
    let mut spectrogram_path = source_path.as_ref().join(relative_path);
    let succeess = spectrogram_path.set_extension("spectrogram");
    assert!(succeess);
    if !spectrogram_path.is_file() {
        log::error!("not a file: {}", spectrogram_path.display());
    }

    let mut width = 0;
    let mut height = 0;
    let spectogram = load_spectrogram(spectrogram_path, &mut width, &mut height)?;

    assert_eq!(width, spectogram_dim[0], "dim missmatch");
    assert_eq!(height, spectogram_dim[1], "dim missmatch");

    let new_point = AnimeDataPoint {
        path: label.path.clone(),
        spectogram,
        expected_outputs: label.expected_outputs(),
    };

    Ok(DataPoint::from_anime_data_point(
        new_point,
        spectogram_dim[0],
        spectogram_dim[1],
    ))
}

impl VirtualTrainingDataset {
    // Returns the number of (in, out) nodes needed in layers
    pub fn get_dimensions(&self) -> (usize, usize) {
        if self.virtual_dataset.len() >= 1 {
            const SPECTOGRAM_WIDTH: usize = 512;
            const SPECTOGRAM_HEIGHT: usize = 512;

            (SPECTOGRAM_WIDTH * SPECTOGRAM_HEIGHT, 2)
        } else {
            (0, 0)
        }
    }

    fn get_training_start_end(&self) -> (usize, usize) {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.virtual_dataset.len() as f64)) as usize;

        (0, traning_data_end)
    }
    fn get_validation_start_end(&self) -> (usize, usize) {
        let traning_data_end: usize =
            (self.thresholds[0] * (self.virtual_dataset.len() as f64)) as usize;
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.virtual_dataset.len() as f64)) as usize;

        (traning_data_end, validadtion_data_end)
    }
    fn get_test_start_end(&self) -> (usize, usize) {
        let validadtion_data_end: usize =
            (self.thresholds[1] * (self.virtual_dataset.len() as f64)) as usize;
        let test_data_end: usize = self.virtual_dataset.len();

        (validadtion_data_end, test_data_end)
    }

    pub fn training_len(&self) -> usize {
        let (start, end) = self.get_training_start_end();
        end - start
    }
    pub fn validation_len(&self) -> usize {
        let (start, end) = self.get_validation_start_end();
        end - start
    }
    pub fn test_len(&self) -> usize {
        let (start, end) = self.get_test_start_end();
        end - start
    }

    pub fn training_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let (data_start, data_end) = self.get_training_start_end();
        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: data_start,
            end: data_end,
        }
    }
    pub fn validation_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let (data_start, data_end) = self.get_validation_start_end();
        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: data_start,
            end: data_end,
        }
    }
    pub fn test_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
        let (data_start, data_end) = self.get_test_start_end();
        VirtualTrainingBatchIter {
            dataset: self,
            batch_size,
            index: data_start,
            end: data_end,
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
