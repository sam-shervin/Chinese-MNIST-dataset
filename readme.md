# Chinese MNIST Classification

This project was implemented two years ago as a practice task for a club recruitment challenge, which I did not apply for but completed out of curiosity. The model achieved an accuracy of **94.46%** on the Chinese MNIST dataset.

---

## Dataset Overview

- **Dataset**: [Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist)
- **Input Shape**: (64, 64, 1) grayscale images
- **Number of Classes**: 15 (corresponding to Chinese digits 0-9 and larger numbers like 100, 1000, etc.)

---

## Data Preprocessing

1. **Label Mapping**: Created a mapping from Chinese numerals to class indices.
2. **One-Hot Encoding**: Converted labels to one-hot encoded vectors for categorical classification.
3. **Image Loading and Processing**:
   - Images were read using OpenCV.
   - Converted to grayscale by averaging RGB channels.
   - Reshaped into (64, 64, 1) format for input into the model.

---

## Model Architecture

The model was implemented using Keras with the following layers:
- **Convolutional Layers**:
  - Conv2D with 32 filters, ReLU activation, followed by MaxPooling.
  - Conv2D with 64 filters, ReLU activation, followed by MaxPooling.
- **Fully Connected Layers**:
  - Flattening layer to convert 2D feature maps to 1D.
  - Dense layer with 128 neurons and ReLU activation.
  - Another Dense layer with 128 neurons and ReLU activation.
- **Output Layer**:
  - Dense layer with 15 neurons and Softmax activation for multi-class classification.

**Model Summary**:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 62, 62, 32)        320       
 max_pooling2d (MaxPooling2D (None, 31, 31, 32)        0         
 conv2d_1 (Conv2D)           (None, 29, 29, 64)        18496     
 max_pooling2d_1 (MaxPooling (None, 14, 14, 64)        0         
 flatten (Flatten)           (None, 12544)             0         
 dense (Dense)               (None, 128)               1605760   
 dense_1 (Dense)             (None, 128)               16512     
 dense_2 (Dense)             (None, 15)                1935      
=================================================================
Total params: 1,643,023
Trainable params: 1,643,023
Non-trainable params: 0
_________________________________________________________________
```

---

## Training Details

- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 10

---

## Results

The model achieved an accuracy of **94.46%** on the test set after 10 epochs, with the following performance:

```
Epoch 1/10
313/313 [==============================] - 6s 6ms/step - loss: 0.7985 - accuracy: 0.7536
Epoch 2/10
313/313 [==============================] - 2s 5ms/step - loss: 0.1856 - accuracy: 0.9414
Epoch 3/10
313/313 [==============================] - 2s 5ms/step - loss: 0.0974 - accuracy: 0.9698
...
```

---

## Key Takeaways

- The model was able to generalize well on the Chinese MNIST dataset with a high accuracy.
- One-hot encoding and appropriate label mapping were crucial for multi-class classification.
- The ConvNet architecture was effective for handwritten character recognition.

---

## Reflections

This project was a great learning experience in:
- Implementing CNNs with Keras.
- Working with image data and preprocessing.
- Experimenting with one-hot encoding for multi-class classification.

Although this was initially done for a club recruitment task I didn't apply for, it was a valuable opportunity to explore deep learning and image classification.

---

## Future Improvements

- Experimenting with different architectures such as ResNet or Inception.
- Using data augmentation to improve generalization.
- Fine-tuning hyperparameters for better performance.

---

## Acknowledgments

- Dataset Source: [Kaggle - Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist)
- Implemented using **Keras** and **OpenCV**.

---

