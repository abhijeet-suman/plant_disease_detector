# Plant Disease Classification System

An interactive deep learning web application powered by **PyTorch** and **Gradio** to detect and classify plant leaf diseases from uploaded images.

---

## 🚀 Features

* **Multi-Image Upload**: Process multiple leaf images simultaneously.
* **Deep Learning Inference**: Uses a pre-trained TorchScript DenseNet model.
* **Interactive UI**: Clean, side-by-side view with thumbnail gallery and confidence predictions.
* **GPU Accelerated**: Optimized for ZeroGPU via Hugging Face.

---

## 📂 Project Structure

```text
├── app.py                     # Main Gradio application
├── plant_model.pt             # TorchScript model weights
├── plant_model.yaml           # Model config, classes, and normalization values
└── requirements.txt           # Project dependencies
