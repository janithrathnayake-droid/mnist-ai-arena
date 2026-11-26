
import joblib

# Load trained models
models = joblib.load("models/mnist_models.joblib")

print("\nModels loaded successfully!\n")

for name, model in models.items():
    print(f"{name}: {type(model).__name__}")

print("\nTotal models:", len(models))
