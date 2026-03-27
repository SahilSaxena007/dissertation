# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
model_stub.py

Tiny Keras model factory whose ONLY purpose is to let joblib / pickle
re-create the SciKeras KerasClassifier inside voting_ensemble.pkl.

The architecture does not need to be identical – it just needs to exist
with the same function name `create_model` and the same input shape (12,).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def create_model():
    """
    Keras model used as a stub for unpickling.

    Input shape: (12,)   # you confirmed nn.model_.input_shape == (None, 12)
    """
    model = Sequential(
        [
            Input(shape=(12,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
