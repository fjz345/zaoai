use rust_mnist::{print_image, Mnist};

pub fn get_mnist() -> Mnist {
    // Load the dataset into an "Mnist" object. If on windows, replace the forward slashes with
    // backslashes.
    let mnist = Mnist::new("mnist/");

    // Print one image (the one at index 5) for verification.
    print_image(&mnist.train_data[5], mnist.train_labels[5]);

    mnist
}
