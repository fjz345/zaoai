use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread::JoinHandle,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::zneural_network::{
    neuralnetwork::NeuralNetwork,
    training::{AIResultMetadata, TrainingSession},
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
    pub payload_buffer: Vec<TrainingThreadPayload>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub handle: Option<JoinHandle<()>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub rx_neuralnetwork: Option<Receiver<NeuralNetwork>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub rx_payload: Option<Receiver<TrainingThreadPayload>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub tx_abort: Option<Sender<()>>,
}

impl TrainingThreadController {
    pub fn begin_training(&mut self, training_session: &TrainingSession) -> bool {
        if let Some(mut nn) = training_session.nn.as_ref() {
            let mut nn = nn.clone();
            let training_data = training_session.training_data.clone();
            let num_epochs = training_session.num_epochs;
            let batch_size = training_session.batch_size;
            let learn_rate = training_session.learn_rate;
            let learn_rate_decay = training_session.learn_rate_decay.clone();
            let learn_rate_decay_rate = training_session.learn_rate_decay_rate;
            let is_correct_fn = training_session.is_correct_fn;

            let (tx_nn, rx_nn) = mpsc::channel();
            let (tx_training_metadata, rx_training_metadata) = mpsc::channel();
            let (tx_abort, rx_abort) = mpsc::channel();

            let training_thread = std::thread::spawn(move || {
                let training_data_vec = training_data.training_split();
                nn.learn(
                    &training_data_vec[..],
                    num_epochs,
                    batch_size,
                    learn_rate,
                    learn_rate_decay,
                    learn_rate_decay_rate,
                    Some(&tx_training_metadata),
                    is_correct_fn,
                    Some(|| rx_abort.try_recv().is_ok()),
                );

                tx_nn.send(nn);
            });

            self.rx_neuralnetwork = Some(rx_nn);
            self.rx_payload = Some(rx_training_metadata);
            self.tx_abort = Some(tx_abort);
            self.payload_buffer = Vec::with_capacity(num_epochs);
            self.handle = Some(training_thread);

            return true;
        }

        false
    }

    pub fn training_in_progress(&self) -> bool {
        if let Some(handle) = &self.handle {
            !handle.is_finished()
        } else {
            false
        }
    }

    pub fn send_abort_training(&self) {
        if let Some(tx) = &self.tx_abort {
            tx.send(());
        }
    }
}
