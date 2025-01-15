# More Bikes: Experiments in Univariate Regression

This repository holds the code and report written to fulfil the coursework
requirements for the unit
[Machine Learning Paradigms](https://www.bris.ac.uk/unit-programme-catalogue/UnitDetails.jsa?ayrCode=23%2F24&unitCode=COMSM0025)
at the University of Bristol.

The assignment was organized as [a Kaggle competition](https://www.kaggle.com/competitions/morebikes2023/overview).
The task was to predict the number of available bikes at 75 rental stations in three hours' time between November 2014 and January 2015,
i.e., a supervised univariate regression problem.

My entry performed best on the private (held-out) and public leaderboards within my PhD cohort.

## Instructions

> [!WARNING]
> This repository is designed to be used on macOS and has not been tested on other operating systems.

### Installation

Create a virtual environment and install Python dependencies:

```bash
conda create --name more-bikes python=3.11
conda activate more-bikes
conda install pip
pip install -r requirements.txt
```

### Execution

To run an experiment or analysis task, run the corresponding VS Code task.
