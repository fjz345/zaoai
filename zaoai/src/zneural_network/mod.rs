pub mod activation;
pub mod cost;
pub mod datapoint;
pub mod is_correct;
pub mod thread;
pub mod training;
pub mod weight_bias;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
