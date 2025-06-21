# TensorFlow, Keras, and the Machine Learning Workflow

---

## What is TensorFlow?

TensorFlow is an open-source framework for machine learning (ML) and artificial intelligence (AI) developed by the **Google Brain** team. It was designed to simplify the development of machine learning models — especially deep learning models — by providing a set of powerful tools to build, train, and deploy models across different platforms.

---

### TensorFlow Architecture

The core idea of TensorFlow is to represent computations as a **computational graph**, where:

- **Tensors**: The basic units of data. These are multi-dimensional arrays (vectors, matrices, or higher-dimensional structures).
- **Graph**: A collection of operations (nodes) and data (edges) that define a complete computation.
- **Session** (in older versions): Responsible for running the computational graph, training models, and making predictions.

In modern TensorFlow (2.x+), eager execution is enabled by default, so code behaves more like standard Python, without requiring explicit sessions.

---

## What is Keras?

Keras is a **high-level deep learning API** that simplifies the process of building deep neural networks.

- Initially developed as an **independent library**, Keras is now tightly integrated into TensorFlow and serves as its **official high-level API**.
- It abstracts away the complexities of tensor operations, making it easier for developers to quickly build and test models.
- Keras supports multiple backend engines, including **TensorFlow**, **Theano**, and **Microsoft Cognitive Toolkit (CNTK)** (although TensorFlow is now the primary and recommended backend).

---

### How to Install Keras

Since Keras is bundled with TensorFlow (2.x), installing TensorFlow installs Keras too:

```bash
pip install tensorflow
```

---

### History of Keras

Keras was developed by **François Chollet**, a Google engineer, as part of a research project called **ONEIROS** (Open-ended Neuro-Electronic Intelligent Robot Operating System). It was released in **March 2015** and quickly became popular due to its simplicity and flexibility.

---

### How to Build a Model in Keras

Keras provides **two main APIs** to define models:

1. **Sequential API**  
   - Best for simple models with a linear stack of layers  
   - Easy to use for models with a single input and single output  

2. **Functional API**  
   - Suitable for complex architectures  
   - Allows multiple inputs, multiple outputs, shared layers, and more flexible connections  

Example using Sequential API:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## The Machine Learning Workflow

Before diving into each step, here’s a quick overview of the standard ML workflow:

### Machine Learning Workflow Steps:

1. Import Libraries  
2. Load and Preprocess Data  
3. Build the Model  
4. Compile the Model  
5. Train the Model  
6. Evaluate the Model  
7. Make Predictions  

---

### 1. Import Libraries

Start by importing essential packages:

- TensorFlow / Keras for model definition and training
- NumPy for handling arrays
- Matplotlib (optional) for visualizing the results

---

### 2. Load and Preprocess Data

- Load the dataset (e.g., MNIST via Keras datasets)
- Normalize input data (e.g., scale pixel values from 0–255 to 0–1)
- Split the data into training and test sets
- Reshape or flatten the data if required by the model

---

### 3. Build the Model

Use Keras’s Sequential or Functional API to build your neural network. A typical model for digit classification might include:

- A Flatten layer to convert the 28x28 image to a 1D vector  
- Dense hidden layers with activation functions (like ReLU)  
- An output layer with softmax activation (for multi-class classification)

---

### 4. Compile the Model

Choose:

- **Optimizer**: e.g., `adam`, `sgd`
- **Loss function**: e.g., `sparse_categorical_crossentropy` for multi-class classification
- **Metrics**: e.g., `accuracy`

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

### 5. Train the Model

Train the model using `.fit()`:

```python
model.fit(x_train, y_train, epochs=5)
```

You can also use a validation split to monitor overfitting.

---

### 6. Evaluate the Model

Use `.evaluate()` on test data to measure how well the model performs on unseen data:

```python
model.evaluate(x_test, y_test)
```

---

### 7. Make Predictions

After training, make predictions with `.predict()` and use functions like `np.argmax()` to extract predicted classes.

```python
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
```

You can compare predictions with true labels and visualize them using Matplotlib.

---

## Summary

TensorFlow and Keras together provide a robust and easy-to-use environment for building deep learning models. With just a few lines of code, you can go from raw data to a working model — and that's the power of a well-structured machine learning workflow.
