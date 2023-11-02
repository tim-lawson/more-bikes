# more-bikes

This repository holds the code and report written to fulfil the coursework
requirements for the unit
[Machine Learning Paradigms](https://www.bris.ac.uk/unit-programme-catalogue/UnitDetails.jsa?ayrCode=23%2F24&unitCode=COMSM0025)
at the University of Bristol.

[The assignment](https://www.kaggle.com/competitions/morebikes2023/overview)
is to predict the number of available bikes at a set of rental stations.
Predictions are evaluated by the mean absolute error (MAE) between the predicted
and true values of the target variable `bikes`.
It is divided into three tasks, which differ in the available data:

1. The number of available bikes at 75 stations over one month
   (`data/train/station_<201–275>_deploy.csv`).

   1. First, train a separate model for each station.
   2. Second, train a single model for all stations.

2. A set of linear models to predict the number of available bikes at 200 other stations
   (`data/models/model_station_<1–200>_<name>.csv`).

3. Both of the above.

# TODO

Analysis and experiments:

- Complete tasks 2 and 3
- Experiment with imputation methods
- Experiment with representations of temporal features
- Constrain models to be non-negative
- Improve implementation of genetic parameter search
- Implement genetic parameter search for task 1.1
- Determine appropriate parameter search spaces
- Experiment with learning algorithms, e.g., kernel regression, Gaussian processes,
  ensemble methods, etc.
- Improve evaluation of learning algorithms and models

Documentation:

- Instructions to install/set up the project
- Instructions to run experiments and tests
- Description of the project structure
- Description of the learning algorithms for each task
- Write the report!
