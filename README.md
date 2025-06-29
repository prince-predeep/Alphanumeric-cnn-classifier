# CNN Handwritten Character Classifier (Digits + Uppercase Letters)

This project is a Convolutional Neural Network (CNN)-based handwritten character recognition system built using TensorFlow and Keras. It classifies grayscale 28x28 images of handwritten characters from **36 classes**: digits (0–9) and uppercase letters (A–Z).

---

## 🧠 What It Does
- **Trains a CNN model** on a filtered subset of 36 character classes (digits + uppercase letters).
- **Classifies custom handwritten images** using OpenCV and matplotlib.
- **Visualizes performance** with classification reports and confusion matrix.

---

## 🧭 Motivation & Process
Originally, the model was trained on the full 62-character dataset (digits, uppercase, and lowercase letters). However, lowercase samples were significantly underrepresented, leading to misclassifications and bias. For a more stable model, only digits and uppercase characters were used.

Key steps and improvements:
- Relabeled dataset to map original labels (0–9, 36–61) to 0–35.
- Centered and normalized custom images for prediction.
- Implemented dropout for regularization and early stopping.

---

## 🧱 Dataset
- **Sources**:
  - Uppercase: *Handwritten A-Z* by Ashish Gupta (Kaggle)
  - Lowercase: *Handwritten English Characters and Digits* by Sujay Mann (Kaggle)
  - Digits: Based on a Kaggle dataset, heavily cleaned and converted to black-on-white for compatibility
- **Format**: CSV with pixel data and labels
- **Classes**: 36 (0–9, A–Z after relabeling)
- **Shape**: 28x28 grayscale pixels

### 📥 Dataset Download
Due to size constraints, the dataset is not included directly in the repository. 

You can download the full preprocessed dataset (CSV format) from this Google Drive link:

👉 **[Download dataset](https://your-google-drive-link-here)**

After downloading, place it in the root folder of this project and rename it to `dataset.csv.txt` if needed.

Alternatively, you can also find it under the [Releases](https://github.com/your-username/your-repo-name/releases) section of this repository.

---

## ⚙️ Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy, pandas, matplotlib, seaborn, OpenCV

Install using:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python
```

---

## 🛠️ How to Run
1. **Prepare your dataset**: Download the CSV and place it in the correct directory.
2. **Train the model**: Run the `main.py` script to train and save your `.keras` model.
3. **Predict from image**:
   - Save handwritten digit/letter images into the `/Captured image` folder.
   - Run the inference section of the script.
   - Output will show predictions via matplotlib.

> ⚠️ **Make sure to update all file paths in the script (dataset, model, image folder, etc.) to match your local directory structure.**

---

## 📌 Notes & Precautions
- Images should be clear and centered. The code auto-detects and centers the digit, but poor lighting or clutter can reduce accuracy.
- The model expects 28x28 grayscale input; resizing and normalization are done automatically.
- Ensure all dependencies are installed and model paths are correct.
- The script assumes the dataset has labels ranging from 0 to 61 (inclusive), but only uses those relabeled to 0–35.

---

## 📁 Folder Structure
```
├── dataset.csv.txt
├── model
│   └── cnn_digitrecog_36class.keras
├── README.md
└── main.py
```

---

## ✅ Output
- **Model Accuracy**: ~88–92% on validation set.
- **Custom Inference**: Predicts characters from user-drawn input images.
- **Visualization**: Confusion matrix and classification report.

---

## 📌 Final Thought
While not state-of-the-art, this classifier demonstrates solid performance across a balanced character set and is ideal for learning and small-scale deployment. Improvements can be made by augmenting the dataset or fine-tuning the CNN.

---

Made with patience, errors, retries... and a little bit of ☕.
