use std::{
    sync::mpsc::{self, Receiver},
    thread::JoinHandle,
};

use serde::{Deserialize, Serialize};

use crate::zneural_network::{
    neuralnetwork::NeuralNetwork,
    training::{AIResultMetadata, TrainingSession},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainingThreadPayload {
    pub payload_index: usize,
    pub payload_max_index: usize,
    pub training_metadata: AIResultMetadata,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingThread {
    pub id: u64,
    pub payload_buffer: Vec<TrainingThreadPayload>,
    #[serde(skip)]
    pub handle: Option<JoinHandle<()>>,
    #[serde(skip)]
    pub rx_neuralnetwork: Option<Receiver<NeuralNetwork>>,
    #[serde(skip)]
    pub rx_payload: Option<Receiver<TrainingThreadPayload>>,
}

impl TrainingThread {
    pub fn new(training_session: TrainingSession) -> Self {
        let nn_option = training_session.nn;
        let training_data = training_session.training_data.clone();
        let num_epochs = training_session.num_epochs;
        let batch_size = training_session.batch_size;
        let learn_rate = training_session.learn_rate;

        let (tx_nn, rx_nn) = mpsc::channel();
        let (tx_training_metadata, rx_training_metadata) = mpsc::channel();

        let training_thread = std::thread::spawn(move || {
            if nn_option.is_some() {
                let mut nn: NeuralNetwork = nn_option.unwrap();

                let training_data_vec = training_data.training_split();
                nn.learn(
                    &training_data_vec[..],
                    num_epochs,
                    batch_size,
                    learn_rate,
                    Some(false),
                    Some(&tx_training_metadata),
                );

                tx_nn.send(nn);
            }
        });

        Self {
            id: 0,
            handle: Some(training_thread),
            rx_neuralnetwork: Some(rx_nn),
            rx_payload: Some(rx_training_metadata),
            payload_buffer: Vec::with_capacity(num_epochs),
        }
    }
}
