"""
Entraîne (par transfer learning) un CNN de classification d'images sur le
jeu de données CIFAR-10, puis sauvegarde le modèle au format SavedModel
pour qu'il soit servi par l'API cloud (app/main.py).

NB : nécessite tensorflow (voir requirements.txt du dossier app/).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
IMG_SIZE = 96  # taille attendue par MobileNetV2


def build_transfer_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # on gèle les poids pré-entraînés

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1.0 / 127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    model = build_transfer_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=64,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision finale sur le jeu de test : {test_acc:.3f}")

    model.save("../app/model/cnn_cifar10_saved_model")
    print("Modèle sauvegardé dans app/model/cnn_cifar10_saved_model/")


if __name__ == "__main__":
    main()
