import keras as K
from keras_tuner import HyperModel
from config import HYPERPARAMETERS_PATH
from utils import read_yaml


class ConvolutionalHyperModel(HyperModel):
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hyperparameters = read_yaml(HYPERPARAMETERS_PATH.format('convolutional'))

    def build(self, hp):
        model = K.models.Sequential()
        model.add(K.layers.InputLayer(input_shape=self.input_shape))
        model.add(K.layers.Rescaling(1. / 255.))

        for i, (filters, activations) in enumerate(zip(self.hyperparameters['filters'], self.hyperparameters['activations'])):

            model.add(K.layers.Conv2D(
                filters=hp.Choice(f'filters_conv{i + 1}', filters),
                kernel_size=hp.Choice(f'kernel_size_conv{i + 1}', self.hyperparameters['kernel_size']),
                strides=hp.Choice(f'strides_conv{i + 1}', self.hyperparameters['strides']),
                activation=hp.Choice(f'activation_conv{i + 1}', activations),
            ))
            model.add(K.layers.MaxPooling2D(pool_size=hp.Choice(f'pool_size_pool{i + 1}', self.hyperparameters['pool_size'])))

        # ---- Feed-Forward Layers ----
        model.add(K.layers.Flatten())

        model.add(K.layers.Dense(
            units=hp.Choice('neurons_dense1', self.hyperparameters['neurons_dense1']),
            activation=hp.Choice('activation_dense1', self.hyperparameters['activations_dense1'])
        )
        )

        model.add(K.layers.Dense(
            units=hp.Choice('neurons_dense2', self.hyperparameters['neurons_dense2']),
            activation=hp.Choice('activation_dense2', self.hyperparameters['activations_dense2'])
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

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            epochs=hp.Choice('epochs', self.hyperparameters['epochs']),
            batch_size=hp.Choice('batch_size', self.hyperparameters['batch_size']),
            **kwargs,
        )
