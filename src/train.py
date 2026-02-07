import os
import tensorflow as tf
from model import build_teeth_model
from preprocess import load_datasets

# 1. Setup Paths & Load Data
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_ds, val_ds, test_ds = load_datasets(base_path)

# 2. Build and Compile
model = build_teeth_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Add Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('outputs/best_model.keras', save_best_only=True)
]

# 4. Train
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15, 
    callbacks=callbacks
)

# 5. Final Evaluation
print("\nEvaluating on Test Set:")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# 6. Save the final version
os.makedirs('outputs', exist_ok=True)
model.save('outputs/teeth_classifier_baseline.keras')