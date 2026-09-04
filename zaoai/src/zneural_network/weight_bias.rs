#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;

use rand_distr::num_traits::*;

use rand::distributions::Distribution;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use zaoai_types::ai_labels::LayerTypeCPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum WeightInit {
    Zero,       // Bad
    Uniform,    // Uniform [0, 1]
    NormalDist, // Normal(0, 1)
    #[default]
    XavierUniform, // sigmoid / tanh
    XavierNormal, // sigmoid / tanh
    HeUniform,  // ReLU / leaky ReLU
    HeNormal,   // ReLU / leaky ReLU
    LeCun,      // SELU / scaled tanh
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum BiasInit {
    Zero,
    #[default]
    ZeroPointZeroOne,
    // Random,
}

pub struct WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: Distribution<T>,
{
    pub weight_init: WeightInit,
    pub _num_inputs: usize,
    pub _num_outputs: usize,
    pub normal_dist: Option<rand_distr::Normal<T>>,
    pub limit: Option<T>,
}

impl<T> WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: rand::distributions::Distribution<T>,
{
    #[inline(always)]
    fn to_t(x: f64) -> T {
        T::from_f64(x).expect("conversion from f64 to T failed")
    }

    pub fn new(weight_init: WeightInit, num_inputs: usize, num_outputs: usize) -> Self {
        let (normal_dist, limit) = match weight_init {
            WeightInit::NormalDist => (
                Some(Normal::new(Self::to_t(0.0), Self::to_t(1.0)).unwrap()),
                None,
            ),
            WeightInit::XavierUniform => {
                let limit = (Self::to_t(6.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (None, Some(limit))
            }
            WeightInit::XavierNormal => {
                let std_dev = (Self::to_t(2.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::HeUniform => {
                let limit = (Self::to_t(6.0) / Self::to_t(num_inputs as f64)).sqrt();
                (None, Some(limit))
            }
            WeightInit::HeNormal => {
                let std_dev = (Self::to_t(2.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::LeCun => {
                let std_dev = (Self::to_t(1.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            _ => (None, None), // Zero and Uniform don't need precalc
        };

        Self {
            weight_init,
            _num_inputs: num_inputs,
            _num_outputs: num_outputs,
            normal_dist,
            limit,
        }
    }

    pub fn sample_weight(&self, rng: &mut ChaCha8Rng) -> T {
        match self.weight_init {
            WeightInit::Zero => Self::to_t(0.0),
            WeightInit::Uniform => {
                T::from_f64(rng.gen_range(0.0..1.0)).expect("Uniform range failed")
            }
            WeightInit::NormalDist
            | WeightInit::XavierNormal
            | WeightInit::HeNormal
            | WeightInit::LeCun => self.normal_dist.as_ref().unwrap().sample(rng),
            WeightInit::XavierUniform | WeightInit::HeUniform => {
                let limit = self.limit.unwrap();
                let val = rng.gen_range(-limit.to_f64().unwrap()..limit.to_f64().unwrap());
                T::from_f64(val).unwrap()
            }
        }
    }
}

impl WeightInit {
    pub fn all() -> &'static [Self] {
        use WeightInit::*;
        &[
            Zero,
            Uniform,
            NormalDist,
            XavierUniform,
            XavierNormal,
            HeUniform,
            HeNormal,
            LeCun,
        ]
    }
}

impl BiasInit {
    pub fn all() -> &'static [Self] {
        use BiasInit::*;
        &[Zero, ZeroPointZeroOne]
    }

    pub fn sample_bias(self) -> LayerTypeCPU {
        match self {
            Self::Zero => 0.0,
            Self::ZeroPointZeroOne => 0.01,
        }
    }
}
