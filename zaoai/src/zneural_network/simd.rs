#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg(feature = "simd")]
use crate::zneural_network::activation::relu_d;

// ============================
// Cost Functions
// ============================
#[cfg(feature = "simd")]
pub fn mse_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    let error = output_activation - expected_activation;
    // 0.5 * error^2
    f32x8::splat(0.5) * error * error
}
#[cfg(feature = "simd")]
pub fn mse_d_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    output_activation - expected_activation
}

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_multiclass_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);
    -expected * clamped.ln()
}
#[cfg(feature = "simd")]
pub fn cross_entropy_loss_multiclass_d_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    // Assumes inputs are after softmax
    predicted - expected
}

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_binary_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);

    -(expected * clamped.ln() + (one - expected) * (one - clamped).ln())
}
#[cfg(feature = "simd")]
pub fn cross_entropy_loss_binary_d_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    // Clamp predicted to [epsilon, 1 - epsilon]
    let p = predicted.min(one - epsilon).max(epsilon);
    let one_minus_p = one - p;
    let one_minus_y = one - expected;

    // Derivative: - y / p + (1 - y) / (1 - p)
    -expected / p + one_minus_y / one_minus_p
}
// ============================

// ============================
// Activation Functions
// ============================
#[cfg(feature = "simd")]
pub fn sigmoid_simd(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + (-x).exp())
}

#[cfg(feature = "simd")]
pub fn sigmoid_d_simd(x: f32x8) -> f32x8 {
    let fx = sigmoid_simd(x);
    fx * (f32x8::splat(1.0) - fx)
}

#[cfg(feature = "simd")]
pub fn relu_simd(in_value: f32x8) -> f32x8 {
    // Fastmax?
    in_value.max(f32x8::splat(0.0))
}

#[cfg(feature = "simd")]
pub fn relu_d_simd(x: f32x8) -> f32x8 {
    // temp fix
    let a: Vec<f32> = x.to_array().iter_mut().map(|f| relu_d(*f)).collect();
    return f32x8::from(&a[..]);

    // think this is correct?, no was wrong...
    // use wide::CmpGt;
    // x.cmp_gt(f32x8::splat(0.0))
}

// ============================
