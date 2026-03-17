import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

DATASET = "alternative_medicine_dataset_balanced_5000.csv"
MODEL_PATH = "model/cnn_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
SCALER_PATH = "model/scaler.pkl"
GRAPH_PATH = "static/images/"

os.makedirs("model", exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

FEATURES = [
    "Age_Group","Gender","Primary_Symptom","Secondary_Symptom",
    "Symptom_Duration_Days","Severity","Chronic_Condition",
    "Stress_Level","Sleep_Quality","Lifestyle","Diet_Type",
    "Previous_Treatment","Dosage_Form","Treatment_Duration","Follow_Up_Required"
]

df = pd.read_csv(DATASET)

# Encode target only
label_encoder = LabelEncoder()
df["Recommended_Treatment"] = label_encoder.fit_transform(df["Recommended_Treatment"])

with open(ENCODER_PATH,"wb") as f:
    pickle.dump(label_encoder,f)

# One-Hot Encoding for features
df = pd.get_dummies(df, columns=FEATURES)

# Keep only numeric columns
X = df.select_dtypes(include=[np.number]).drop("Recommended_Treatment", axis=1)
y = df["Recommended_Treatment"]

# Train/Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open(SCALER_PATH,"wb") as f:
    pickle.dump(scaler,f)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# CNN Model
model = Sequential([

    Conv1D(32,3,activation="relu",padding="same",input_shape=(X_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64,3,activation="relu",padding="same"),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(len(np.unique(y)),activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model (20 epochs as required)
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# Predictions
y_pred = np.argmax(model.predict(X_test),axis=1)

acc = accuracy_score(y_test,y_pred)

# Save model
model.save(MODEL_PATH)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(GRAPH_PATH+"confusion_matrix.png")
plt.close()

# Training Graph
plt.figure()
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.title("CNN Training Performance")
plt.savefig(GRAPH_PATH+"training_graphs.png")
plt.close()

print("Final Accuracy:",round(acc*100,2))