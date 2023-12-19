"""Task 1B: Multi-layer perceptron."""

from numpy import nan
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.preprocessing.column import column_transformer_1b
from more_bikes.preprocessing.ordinal import ordinal_transformer
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

params = [
    {
        "regressor__mlpregressor__hidden_layer_sizes": [
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
        ],
        "regressor__mlpregressor__activation": [
            "logistic",
            # "tanh",
            # "relu",
        ],
        "regressor__mlpregressor__learning_rate": [
            # "constant",
            "invscaling",
            "adaptive",
        ],
    }
]


def mlp():
    """Multi-layer perceptron."""
    return Task1BExperiment(
        model=Model(
            name="mlp",
            pipeline=TransformedTargetRegressor(
                make_pipeline(
                    ordinal_transformer.set_output(transform="pandas"),
                    column_transformer_1b.set_output(transform="pandas"),
                    StandardScaler(),
                    SimpleImputer(missing_values=nan, strategy="mean"),
                    MLPRegressor(random_state=42),
                ),
                transformer=BikesFractionTransformer(),
            ),
            params=params,
        ),
    )


if __name__ == "__main__":
    mlp().run().save()
