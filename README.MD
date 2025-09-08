# ft_linear_regression

## Overview

`ft_linear_regression` is a Python-based project that implements a simple linear regression model. The project includes functionality to train the model on a dataset, save the learned parameters, and make predictions using the trained model.

## Features

- Train a linear regression model on a dataset.
- Save and load model parameters (`thetas.json`).
- Predict outcomes based on input data.
- Modular design with utility functions for data handling.

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv          # Dataset for training the model
├── predict.py            # Script to make predictions
├── train.py              # Script to train the model
├── utils.py              # Utility functions for data handling
├── requirements.txt      # Python dependencies
├── Makefile              # Makefile for automation
├── README.MD             # Project documentation
```

## Prerequisites

- Python 3.10 or higher
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bac0nb0yy/ft_linear_regression.git
   cd ft_linear_regression
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model using the dataset in `data/data.csv`:

```bash
python train.py
```

The trained parameters will be saved in `thetas.json`.

### Making Predictions

To make predictions using the trained model:

```bash
python predict.py
```

Ensure that `thetas.json` exists and contains the trained parameters.

## File Descriptions

- **train.py**: Contains the logic for training the linear regression model.
- **predict.py**: Uses the trained model to make predictions.
- **utils.py**: Includes helper functions for data processing and other utilities.
- **thetas.json**: Stores the model parameters (weights and bias).
- **data/data.csv**: Example dataset for training the model.

## Makefile

The `Makefile` includes automation for common tasks:

- `make`: Run the training then you can predict.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.
