import joblib
import inspect
from model_stub import create_model


path = "../artifacts/voting_ensemble.pkl"

print("\n🔍 Loading pickle…")
data = joblib.load(path)

print("\n🔑 Keys found:", data.keys())

print("\n📌 Types:")
for key, value in data.items():
    print(f"• {key} → {type(value)}")

print("\n🎯 Checking if Keras model is inside…")
for key, value in data.items():
    try:
        print(key, value.__class__.__module__)
    except:
        pass

models = joblib.load("../artifacts/voting_ensemble.pkl")
nn = models["nn"]
print('jdfjdfjdfn: ',nn.model_.input_shape)
