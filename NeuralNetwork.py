import os
from pathlib import Path
import pickle
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_FILE = "hazard_dummy_data_v2.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SCALER_PATH = MODELS_DIR / "scaler_and_encoders.pkl"
FIRE_MODEL_PATH = MODELS_DIR / "fire_risk_regressor.h5"
FLOOD_MODEL_PATH = MODELS_DIR / "flood_risk_regressor.h5"
CLASS_MODEL_PATH = MODELS_DIR / "hazard_classifier.h5"
PLOT_PATH = "training_plot.png"

if not Path(DATA_FILE).exists():
    print(f"{DATA_FILE} not found. Generating dataset using generate_csv.py ...")
    spec = importlib.util.spec_from_file_location("generate_csv", "generate_csv.py")
    generate_csv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_csv)
    df = generate_csv.generate_dataset(n_samples=200)
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset saved to {DATA_FILE}")
else:
    df = pd.read_csv(DATA_FILE)

print("Loaded dataset:", DATA_FILE)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# Regression targets: fire_risk_score, flood_risk_score (continuous)
# Classification target: hazard_class (low/moderate/high)
reg_targets = ["fire_risk_score", "flood_risk_score"]
class_target = "hazard_class"

if not set(reg_targets + [class_target]).issubset(df.columns):
    raise ValueError("Required target columns missing from dataset.")

# Drop columns that are unnecessary or identifiers
drop_cols = ["town_name", "wind_direction"]  # we'll encode wind_direction, but keep in case - we will encode it
# We'll use these features:
candidate_features = [
    'latitude', 'longitude', 'population_density', 'urbanization_index',
    'temperature', 'humidity', 'precipitation_last_24h', 'precipitation_last_7d',
    'wind_speed', 'days_since_last_rain', 'avg_temp_past_week', 'max_temp_past_week',
    'elevation', 'slope', 'aspect', 'distance_to_water', 'impervious_surface_ratio',
    'vegetation_density', 'soil_type', 'soil_moisture', 'surface_runoff',
    'streamflow_index', 'drought_index', 'storm_warning_flag', 'recent_fire_flag'
]

# Keep only columns that exist
features = [c for c in candidate_features if c in df.columns]

# Prepare X
X_raw = df[features].copy()

# Categorical columns to one-hot encode
cat_cols = []
if 'soil_type' in X_raw.columns:
    cat_cols.append('soil_type')
# wind_direction was dropped above but if present in CSV, you can include it:
if 'wind_direction' in df.columns and 'wind_direction' not in X_raw.columns:
    # add wind_direction if you want it
    X_raw['wind_direction'] = df['wind_direction']
    cat_cols.append('wind_direction')

num_cols = [c for c in X_raw.columns if c not in cat_cols]

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# Fill NAs
X_raw[num_cols] = X_raw[num_cols].fillna(0)
for c in cat_cols:
    X_raw[c] = X_raw[c].fillna("MISSING")

# Encode categorical -> OneHotEncoder
ohe = None
if cat_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_arrays = ohe.fit_transform(X_raw[cat_cols])
    cat_columns = list(ohe.get_feature_names_out(cat_cols))
    X_num = X_raw[num_cols].to_numpy(dtype=float)
    X = np.hstack([X_num, cat_arrays])
    feature_names = num_cols + cat_columns
else:
    X = X_raw[num_cols].to_numpy(dtype=float)
    feature_names = num_cols

print("Final feature count:", X.shape[1])

# Encode classification labels
label_enc = LabelEncoder()
y_class = label_enc.fit_transform(df[class_target].astype(str))  # 0/1/2

# Regression targets
y_fire = df["fire_risk_score"].values.astype(float)
y_flood = df["flood_risk_score"].values.astype(float)

# Split data
X_train, X_test, y_fire_train, y_fire_test, y_flood_train, y_flood_test, y_class_train, y_class_test = \
    train_test_split(X, y_fire, y_flood, y_class, test_size=0.2, random_state=42, shuffle=True)

# Scale numeric inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler + encoders
with open(SCALER_PATH, "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "ohe": ohe,
        "label_enc": label_enc,
        "feature_names": feature_names
    }, f)
print("Saved scaler and encoders to", SCALER_PATH)

input_dim = X_train.shape[1]
inputs = Input(shape=(input_dim,), name="inputs")

# Shared representation
x = layers.Dense(128, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Two regression heads
fire_out = layers.Dense(32, activation="relu")(x)
fire_out = layers.Dense(1, activation="sigmoid", name="fire_risk")(fire_out)  # 0-1

flood_out = layers.Dense(32, activation="relu")(x)
flood_out = layers.Dense(1, activation="sigmoid", name="flood_risk")(flood_out)  # 0-1

# Classification head (3 classes)
clf_x = layers.Dense(64, activation="relu")(x)
clf_x = layers.Dropout(0.2)(clf_x)
clf_out = layers.Dense(3, activation="softmax", name="hazard_class")(clf_x)

model = Model(inputs=inputs, outputs=[fire_out, flood_out, clf_out], name="hazard_multioutput")
model.summary()

# Compile - multi-loss weighting: regression use MSE/MAE, classification use sparse categorical crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "fire_risk": "mse",
        "flood_risk": "mse",
        "hazard_class": "sparse_categorical_crossentropy"
    },
    loss_weights={"fire_risk": 1.0, "flood_risk": 1.0, "hazard_class": 1.0},
    metrics={
        "fire_risk": ["mae"],
        "flood_risk": ["mae"],
        "hazard_class": ["accuracy"]
    }
)

early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)

# Train
history = model.fit(
    X_train,
    {"fire_risk": y_fire_train, "flood_risk": y_flood_train, "hazard_class": y_class_train},
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

model.save(MODELS_DIR / "hazard_multioutput.h5")
print("Saved multi-output model to models/hazard_multioutput.h5")

# If you prefer separate models, you can create & save splits of this model's outputs — below are convenience slices:
# Save separate small models for serving:
fire_model = Model(inputs=inputs, outputs=fire_out, name="fire_model")
flood_model = Model(inputs=inputs, outputs=flood_out, name="flood_model")
class_model = Model(inputs=inputs, outputs=clf_out, name="class_model")

fire_model.save(FIRE_MODEL_PATH)
flood_model.save(FLOOD_MODEL_PATH)
class_model.save(CLASS_MODEL_PATH)
print("Saved individual models to:", FIRE_MODEL_PATH, FLOOD_MODEL_PATH, CLASS_MODEL_PATH)

pred_fire_test, pred_flood_test, pred_class_test = model.predict(X_test)

# Regression metrics
mae_fire = mean_absolute_error(y_fire_test, pred_fire_test.flatten())
mse_fire = mean_squared_error(y_fire_test, pred_fire_test.flatten())
mae_flood = mean_absolute_error(y_flood_test, pred_flood_test.flatten())
mse_flood = mean_squared_error(y_flood_test, pred_flood_test.flatten())

print("\n===== REGRESSION METRICS =====")
print(f"Fire - MAE: {mae_fire:.4f}, MSE: {mse_fire:.4f}")
print(f"Flood - MAE: {mae_flood:.4f}, MSE: {mse_flood:.4f}")

# Classification metrics
pred_class_labels = np.argmax(pred_class_test, axis=1)
acc = accuracy_score(y_class_test, pred_class_labels)
print("\n===== CLASSIFICATION METRICS =====")
print("Accuracy:", acc)
print(classification_report(y_class_test, pred_class_labels, target_names=label_enc.classes_))

plt.figure(figsize=(12, 6))

#Total loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.plot(epochs, [0.0]*len(epochs), '--', color='gray', label='Ideal (0)')
plt.title('Total Loss')
plt.xlabel('Epochs')
plt.legend()

#Classification accuracy
# TensorFlow may store as 'hazard_class_accuracy' or 'hazard_class_acc'
possible_keys = ['hazard_class_accuracy', 'hazard_class_acc', 'val_hazard_class_accuracy', 'val_hazard_class_acc']

# dynamically pick training/validation keys
train_key = next((k for k in history.history.keys() if 'hazard_class' in k and 'val' not in k), None)
val_key = next((k for k in history.history.keys() if 'hazard_class' in k and 'val' in k), None)

acc_hist = history.history[train_key] if train_key else []
val_acc_hist = history.history[val_key] if val_key else []

plt.subplot(1, 2, 2)
if acc_hist:
    plt.plot(range(1, len(acc_hist)+1), acc_hist, label='Train Hazard Class Acc')
    plt.plot(range(1, len(val_acc_hist)+1), val_acc_hist, label='Val Hazard Class Acc')
plt.plot(epochs, [1.0]*len(epochs), '--', color='gray', label='Ideal (1.0)')
plt.title('Classification Accuracy (hazard_class)')
plt.xlabel('Epochs')
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)

print("Saved training plot to", PLOT_PATH)
