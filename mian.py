# ✅ Filtered CNN Training with Relabeled Classes (Digits + Uppercase Only)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# ✅ Step 1: Load the full dataset
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Programs\Projects\ML Digitrecog\dataset.csv.txt")

# ✅ Step 2: Keep only digits (0–9) and uppercase letters (36–61)
filtered_data = data[(data['label'] <= 9) | (data['label'] >= 36)].copy()

# ✅ Step 3: Remap labels (0–9 stay the same, 36–61 becomes 10–35)
filtered_data['label'] = filtered_data['label'].apply(lambda x: x if x <= 9 else x - 26)

# ✅ Step 4: Extract features and labels
X = filtered_data.drop('label', axis=1).values.astype(np.float32)
y = filtered_data['label'].values.astype(np.int32)

# ✅ Step 5: Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# ✅ Step 6: Train-test split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Step 7: One-hot encode labels
num_classes = 36
train_y_cat = tf.keras.utils.to_categorical(train_y, num_classes)
test_y_cat = tf.keras.utils.to_categorical(test_y, num_classes)

# ✅ Step 8: Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ✅ Step 9: Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_x, train_y_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ✅ Step 10: Evaluate and save
loss, acc = model.evaluate(test_x, test_y_cat)
print(f"Test Accuracy: {acc * 100:.2f}%")

model.save(r"C:\Users\LENOVO\Desktop\Programs\Projects\ML Digitrecog\model\cnn_digitrecog_36class.keras")

# ✅ Step 11: Classification Report & Confusion Matrix
pred_y = model.predict(test_x)
pred_labels = np.argmax(pred_y, axis=1)
true_labels = np.argmax(test_y_cat, axis=1)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='mako')
plt.title("Confusion Matrix (Digits + Uppercase Only)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Inference: Custom Image Prediction
model = load_model(r"C:\Users\LENOVO\Desktop\Programs\Projects\ML Digitrecog\model\cnn_digitrecog_36class.keras")

label_map = {i: chr(i + 48) if i < 10 else chr(i + 55) for i in range(36)}  # 0-9, A-Z
img_folder = r"C:\Users\LENOVO\Desktop\img"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    resized = cv2.resize(digit, (20, 20))
    padded = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    normalized = padded / 255.0
    final_input = normalized.reshape(1, 28, 28, 1)
    return final_input, padded

for img_name in os.listdir(img_folder):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(img_folder, img_name)
        input_data, display_img = preprocess_image(img_path)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        predicted_label = label_map.get(predicted_class, str(predicted_class))

        plt.imshow(display_img, cmap='gray')
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
