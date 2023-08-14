// use mnist::*;
// use ndarray::{prelude::*, OwnedRepr};

// pub fn get_mnist(
//     train_data: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
//     train_labels: &mut Array2<f32>,
// ) {
//     // Deconstruct the returned Mnist struct.
//     let Mnist {
//         trn_img,
//         trn_lbl,
//         tst_img,
//         tst_lbl,
//         ..
//     } = MnistBuilder::new()
//         .label_format_digit()
//         .training_set_length(50_000)
//         .validation_set_length(10_000)
//         .test_set_length(10_000)
//         .finalize();

//     let image_num = 0;
//     // Can use an Array2 or Array3 here (Array3 for visualization)
//     *train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
//         .expect("Error converting images to Array3 struct")
//         .map(|x| *x as f32 / 256.0);
//     println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

//     // Convert the returned Mnist struct to Array2 format
//     *train_labels = Array2::from_shape_vec((50_000, 1), trn_lbl)
//         .expect("Error converting training labels to Array2 struct")
//         .map(|x| *x as f32);
//     println!(
//         "The first digit is a {:?}",
//         train_labels.slice(s![image_num, ..])
//     );

//     let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
//         .expect("Error converting images to Array3 struct")
//         .map(|x| *x as f32 / 256.);

//     let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
//         .expect("Error converting testing labels to Array2 struct")
//         .map(|x| *x as f32);
// }

use rust_mnist::{print_image, Mnist};

pub fn get_mnist() {
    // Load the dataset into an "Mnist" object. If on windows, replace the forward slashes with
    // backslashes.
    let mnist = Mnist::new("mnist/");

    // Print one image (the one at index 5) for verification.
    print_image(&mnist.train_data[5], mnist.train_labels[5]);
    println!("Mnist...");
}
