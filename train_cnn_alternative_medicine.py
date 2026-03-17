# ============================================
# High-Accuracy CNN Model for Alternative Medicine Recommendation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============================================
# Load Dataset
# ============================================
df = pd.read_csv("alternative_medicine_dataset_patterned_5000.csv")
print("Dataset Loaded:", df.shape)

# ============================================
# Encode Categorical Features
# ============================================
categorical_cols = [
    "Age_Group", "Gender",
    "Primary_Symptom", "Secondary_Symptom",
    "Severity", "Chronic_Condition",
    "Stress_Level", "Sleep_Quality",
    "Lifestyle", "Diet_Type",
    "Previous_Treatment",
    "Dosage_Form", "Treatment_Duration",
    "Follow_Up_Required"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["Recommended_Treatment"] = target_encoder.fit_transform(
    df["Recommended_Treatment"]
)

# ============================================
# Feature Selection (Refined)
# ============================================
X = df.drop(columns=[
    "Patient_ID",
    "Consultation_Date",
    "Recommended_Treatment"
])

y = df["Recommended_Treatment"]

# ============================================
# Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ============================================
# Scaling
# ============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# Reshape for CNN
# ============================================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ============================================
# Build Improved CNN Model
# ============================================
# ============================================
# Build SAFE High-Accuracy CNN Model
# ============================================
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same',
           input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()

# ============================================
# Callbacks
# ============================================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)

# ============================================
# Train Model
# ============================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================
# Evaluate Model
# ============================================
y_pred = np.argmax(model.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# Confusion Matrix
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - CNN Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
# Training Curves
# ============================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ============================================
# Save Model
# ============================================
model.save("cnn_alternative_medicine_model.h5")
print("\nModel saved as cnn_alternative_medicine_model.h5")
