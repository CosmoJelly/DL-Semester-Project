# model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from config import IMG_SIZE, LEARNING_RATE

def get_base_model(name):
    size = (IMG_SIZE[0], IMG_SIZE[1], 3)
    name = name.lower()
    if name == "densenet201":
        return tf.keras.applications.DenseNet201(weights="imagenet", include_top=False, input_shape=size)
    if name == "densenet121":
        return tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=size)
    if name == "inceptionresnetv2":
        return tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=size)
    if name == "inceptionv3":
        return tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=size)
    if name == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=size)
    if name == "nasnetlarge":
        return tf.keras.applications.NASNetLarge(weights="imagenet", include_top=False, input_shape=size)
    if name == "nasnetmobile":
        return tf.keras.applications.NASNetMobile(weights="imagenet", include_top=False, input_shape=size)
    if name == "resnet152v2":
        return tf.keras.applications.ResNet152V2(weights="imagenet", include_top=False, input_shape=size)
    if name == "vgg19":
        return tf.keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=size)
    if name == "xception":
        return tf.keras.applications.Xception(weights="imagenet", include_top=False, input_shape=size)
    raise ValueError("Unsupported model name: " + name)

def build_model(name, num_classes, base_trainable=False, dropout=0.4):
    base = get_base_model(name)
    base.trainable = base_trainable
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(base.input, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model