import keras as K
from keras_tuner import HyperModel
from config import HYPERPARAMETERS_PATH
from utils import read_yaml


class FeedForwardHyperModel(HyperModel):
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hyperparameters = read_yaml(HYPERPARAMETERS_PATH.format('feed_forward'))

    def build(self, hp):
        model = K.models.Sequential()
        model.add(K.layers.Flatten(input_shape=self.input_shape))
        model.add(K.layers.Rescaling(1. / 255.))
        for units, activations in zip(self.hyperparameters['units'], self.hyperparameters['activations']):
            model.add(
                K.layers.Dense(
                    units=hp.Choice('units', units),
                    activation=hp.Choice('activation', activations),
                )
            )

        model.add(K.layers.Dense(self.num_classes, activation='softmax', use_bias=hp.Boolean('use_bias')))

        optimizer = K.optimizers.legacy.Adam(
            hp.Choice('learning_rate', self.hyperparameters['learning_rate'])
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            epochs=hp.Choice('epochs', self.hyperparameters['epochs']),
            batch_size=hp.Choice('batch_size', self.hyperparameters['batch_size']),
            ** kwargs,
        )
