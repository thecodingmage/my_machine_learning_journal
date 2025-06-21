# MNIST Handwritten Digit Classifier

This project implements a simple neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Project Structure

```
mnist_handwritten_digit_classifier/
├── README.md           # Project documentation
├── mnist_model.py      # Model building and training script
```



## About the Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0 to 9), each of size 28x28 pixels. It is a standard benchmark dataset for image classification.

## Model Overview

- Input: 28x28 grayscale images
- Architecture: Sequential Neural Network
  - Flatten layer to convert 2D input to 1D
  - Dense hidden layer with ReLU activation
  - Output layer with Softmax activation for classification
- Output: Probability distribution over 10 classes (digits 0–9)

## Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib (optional for visualization)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/mnist_handwritten_digit_classifier

2. Install dependencies:
    pip install tensorflow numpy matplotlib

3. Run the model:
    python mnist_model.py


## Results

The model achieves approximately 98% accuracy on the test set after training for a few epochs.

## Author

Nakibul Islam
