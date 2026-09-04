pub mod layer;
pub mod neuralnetwork_cpu;
#[cfg(not(feature = "simd"))]
pub mod scalar;
#[cfg(feature = "simd")]
pub mod simd;
