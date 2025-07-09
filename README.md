# zaoai
Z Anime Opening AI

![alt text](img/showcase2.png)

# This project is still in its infancy stage
Features at this moment:
    * Create neural networks with any graph layout
    * Train that network using backpropegation
    * Provides a interactive UI
    * SIMD optimized (30 epochs for [784, 100, 10] network takes a few seconds locally)

# Goal of this application:
Input a anime video file, add chapter timestamps for it for OP start/end & ED start/end.
This will be done by analyzing mainly the audio.

# Features TODO list:
Support Audio Formats
    * mkv
    * mp4

## TODO:
    Create ML AI
        * Save/load weights
        * Dropout neurons
        * Cross-validation
        * Eigen vectors? (reduce amount of input tensors)
        * Gradient dedcent momentum & decay
        * Add noise
    Add chapters to video file
    Gather training data
    Figure out how to train AI