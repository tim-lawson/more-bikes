"""Task 1B: Multi-layer perceptron."""

from numpy import nan
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from more_bikes.experiments.experiment import Model
from more_bikes.experiments.task_1b.task_1b_experiment import Task1BExperiment
from more_bikes.feature_selection.variance_threshold import feature_selection_variance
from more_bikes.preprocessing.drop import make_preprocessing_drop
from more_bikes.preprocessing.ordinal import preprocessing_ordinal
from more_bikes.util.processing import BikesFractionTransformer
from more_bikes.util.target import TransformedTargetRegressor

params = [
    {
        "regressor__mlpregressor__hidden_layer_sizes": [
            # (16, 16, 16),
            # (32, 32, 32),
            (64, 64, 64),
        ],
        "regressor__mlpregressor__activation": [
            "logistic",
            # "tanh",
            # "relu",
        ],
        "regressor__mlpregressor__learning_rate": [
            # "constant",
            # "invscaling",
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
                    # preprocessing
                    preprocessing_ordinal,
                    SimpleImputer(strategy="constant", fill_value=0),
                    StandardScaler(),
                    # feature selection
                    feature_selection_variance,
                    make_preprocessing_drop(
                        [
                            "bikes_3h_diff_avg_short",
                            "bikes_avg_short",
                            "wind_speed_avg",
                        ]
                    ),
                    # regression
                    MLPRegressor(random_state=42),
                ),
                transformer=BikesFractionTransformer(),
            ),
            params=params,
        ),
    )
