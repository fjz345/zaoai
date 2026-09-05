use std::{
    sync::mpsc::{self, Receiver, SendError, Sender},
    thread::JoinHandle,
    time::Instant,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    app::NeuralNetworkType,
    zneural_network::training::{AIResultMetadata, TrainingSession},
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct TrainingThreadPayload {
    pub payload_index: usize,
    pub payload_max_index: usize,
    pub training_metadata: AIResultMetadata,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub struct TrainingThreadController {
    pub id: u64,
    pub payload_training_buffer: Vec<TrainingThreadPayload>,
    pub payload_validation_buffer: Vec<TrainingThreadPayload>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub handle: Option<JoinHandle<()>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub rx_neuralnetwork: Option<Receiver<NeuralNetworkType>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub rx_training_payload: Option<Receiver<TrainingThreadPayload>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub rx_validation_payload: Option<Receiver<TrainingThreadPayload>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub tx_abort: Option<Sender<()>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub start_time: Option<Instant>,
}

impl TrainingThreadController {
    pub fn begin_training(&mut self, training_session: &TrainingSession) -> bool {
        let nn = match training_session.nn.as_ref() {
            Some(nn) => nn.clone(),
            None => {
                log::error!("begin_training failed, no NN");
                return false;
            }
        };

        let training_data = training_session.training_data.clone();
        let num_epochs = training_session.num_epochs;
        let batch_size = training_session.batch_size;
        let learn_rate = training_session.learn_rate;
        let learn_rate_decay = training_session.learn_rate_decay.clone();
        let learn_rate_decay_rate = training_session.learn_rate_decay_rate;
        let is_correct_fn = training_session.is_correct_fn;
        let validation_each_epoch = training_session.validation_each_epoch;

        let (tx_nn, rx_nn): (Sender<NeuralNetworkType>, Receiver<NeuralNetworkType>) =
            mpsc::channel();

        let (tx_training_metadata, rx_training_metadata) = mpsc::channel();
        let (tx_validation_metadata, rx_validation_metadata) = mpsc::channel();
        let (tx_abort, rx_abort) = mpsc::channel();

        self.start_time = Some(Instant::now());

        let training_thread = match nn {
            NeuralNetworkType::CPU(mut neural_network_cpu) => std::thread::spawn(move || {
                let training_data_vec = training_data.training_split();
                let validation_data_vec = training_data.validation_split();

                neural_network_cpu.learn(
                    &training_data_vec[..],
                    &validation_data_vec[..],
                    num_epochs,
                    batch_size,
                    learn_rate,
                    learn_rate_decay,
                    learn_rate_decay_rate,
                    Some(&tx_training_metadata),
                    Some(&tx_validation_metadata),
                    is_correct_fn,
                    Some(|| rx_abort.try_recv().is_ok()),
                    validation_each_epoch,
                );

                if let Err(e) = tx_nn.send(NeuralNetworkType::CPU(neural_network_cpu)) {
                    log::error!("Failed to send neural network through channel: {}", e);
                }
            }),

            NeuralNetworkType::GPU(mut neural_network_gpu) => std::thread::spawn(move || {
                let training_data_vec = training_data.training_split();
                let validation_data_vec = training_data.validation_split();

                neural_network_gpu.learn(
                    &training_data_vec[..],
                    &validation_data_vec[..],
                    num_epochs,
                    batch_size,
                    learn_rate,
                    learn_rate_decay,
                    learn_rate_decay_rate,
                    Some(&tx_training_metadata),
                    Some(&tx_validation_metadata),
                    is_correct_fn,
                    Some(|| rx_abort.try_recv().is_ok()),
                    validation_each_epoch,
                );

                if let Err(e) = tx_nn.send(NeuralNetworkType::GPU(neural_network_gpu)) {
                    log::error!("Failed to send neural network through channel: {}", e);
                }
            }),
        };

        self.rx_neuralnetwork = Some(rx_nn);
        self.rx_training_payload = Some(rx_training_metadata);
        self.rx_validation_payload = Some(rx_validation_metadata);
        self.tx_abort = Some(tx_abort);

        self.payload_training_buffer = Vec::with_capacity(num_epochs + 1);
        self.payload_validation_buffer = Vec::with_capacity(num_epochs + 1);

        self.handle = Some(training_thread);

        true
    }

    pub fn training_in_progress(&self) -> bool {
        if let Some(handle) = &self.handle {
            !handle.is_finished()
        } else {
            false
        }
    }

    pub fn send_abort_training(&self) -> Result<(), SendError<()>> {
        if let Some(tx) = &self.tx_abort {
            tx.send(())?;
        }
        Ok(())
    }
}
