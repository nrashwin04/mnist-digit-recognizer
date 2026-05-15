# 📝Handwritten Digit Recognizer

A PyTorch-based Convolutional Neural Network (CNN) built to recognize handwritten digits using the MNIST dataset. Includes an interactive web interface built with Streamlit!

## Features
- **Custom CNN Architecture**: Uses 2 Conv2d layers, MaxPooling, BatchNorm2d, Dropout, and 2 Fully Connected layers.
- **Interactive UI**: Draw digits directly on a canvas or upload photos.
- **Smart Preprocessing**: Automatically centers, pads, and autocontrasts real-world images to match MNIST dataset standards.
- **Detailed Analytics**: Outputs a 10-digit probability bar chart and confidence percentage.
- **Performance Evaluation**: Generates sklearn Classification Reports and Seaborn Confusion Matrices.

## Live Demo
 
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/niwssa/explainable-ml-dashboard)
 
## How to Run Locally

1. Install dependencies:
   ```bash
   pip install torch torchvision Pillow streamlit plotly pandas streamlit-drawable-canvas numpy scikit-learn seaborn matplotlib onnx
   ```

2. Start the interactive web interface:
   ```bash
   streamlit run app.py
   ```

3. (Optional) Retrain the model manually:
   ```bash
   python mnist_cnn.py
   ```
   *This will run through 5 epochs, save the weights to `mnist_cnn.pth`, and export the architecture to `mnist_cnn.onnx`.*

## Project Structure
- `app.py`: The Streamlit web interface.
- `mnist_cnn.py`: The core PyTorch neural network, training loop, and image preprocessing functions.
- `requirements.txt`: Deployment dependencies.
