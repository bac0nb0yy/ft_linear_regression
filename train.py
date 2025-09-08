import json
import math
import signal
import sys
from argparse import ArgumentParser
from random import randint

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from predict import estimate_price
from utils import CSVValidationError, validate_csv_structure

ERROR_FUNCTIONS = {"MAE": abs, "MSE": lambda x: x * x}


def normalize(value, mean, std):
    return (value - mean) / std


def mean(values):
    return sum(values) / len(values)


def variance(values, mean):
    return sum((x - mean) ** 2 for x in values) / len(values)


def standard_deviation(values, mean):
    return math.sqrt(variance(values, mean))


def train(data, l_rate, epochs, error_func, verbose):
    theta1 = theta0 = 0

    mileages_denormalize = data["km"].tolist()
    prices_denormalize = data["price"].tolist()

    prices_mean = mean(prices_denormalize)
    mileages_mean = mean(mileages_denormalize)
    prices_std = standard_deviation(prices_denormalize, prices_mean)
    mileages_std = standard_deviation(mileages_denormalize, mileages_mean)
    prices = [
        normalize(p, prices_mean, prices_std) for p in prices_denormalize
    ]
    mileages = [
        normalize(m, mileages_mean, mileages_std) for m in mileages_denormalize
    ]
    m = len(data)

    for epoch in tqdm(range(epochs)):
        tmp0 = l_rate * (
            sum(
                estimate_price(theta1, theta0, mileages[i]) - prices[i]
                for i in range(m)
            )
            / m
        )
        tmp1 = l_rate * (
            sum(
                (estimate_price(theta1, theta0, mileages[i]) - prices[i])
                * mileages[i]
                for i in range(m)
            )
            / m
        )
        theta1 -= tmp1
        theta0 -= tmp0

        if verbose:
            C = theta1 * (prices_std / mileages_std)
            error = mean(
                [
                    ERROR_FUNCTIONS[error_func](
                        estimate_price(
                            C,
                            prices_mean - C * mileages_mean,
                            km,
                        )
                        - price
                    )
                    for price, km in zip(data["price"], data["km"])
                ]
            )
            print(
                f"Average {error:.2f} off real prices, with error function {error_func} after {epoch} epoch(s)."
            )

    theta1 = theta1 * (prices_std / mileages_std)
    theta0 = (theta0 * prices_std) - (theta1 * mileages_mean) + prices_mean
    error = mean(
        [
            ERROR_FUNCTIONS[error_func](
                estimate_price(theta1, theta0, km) - price
            )
            for price, km in zip(data["price"], data["km"])
        ]
    )
    print(
        f"Average {error:.2f} off real prices, with error function {error_func} after {epochs} epochs."
    )
    return theta1, theta0


def plot_result(data, theta1, theta0):
    y_line = [(theta1 * x + theta0) for x in data["km"]]

    plt.figure("Linear Regression Results", figsize=(12, 6))

    plt.scatter(
        data["km"],
        data["price"],
        color="blue",
        alpha=0.6,
        edgecolor="k",
        s=50,
        label="Actual Prices",
    )

    plt.plot(
        data["km"],
        y_line,
        color="red",
        linewidth=2,
        label=f"Regression Line: y = {theta1:.2f}x + {theta0:.2f}",
    )

    plt.title("Price Prediction based on Kilometers Driven", fontsize=16)
    plt.xlabel("Kilometers Driven (km)", fontsize=14)
    plt.ylabel("Price", fontsize=14)

    plt.grid(True)
    plt.legend()
    plt.xlim(0, max(data["km"]) * 1.05)
    plt.ylim(0, max(data["price"]) * 1.05)

    mid_x = data["km"].iloc[randint(0, len(data["km"]) - 1)]
    mid_y = estimate_price(theta1, theta0, mid_x)
    plt.annotate(
        f"({mid_x}, {mid_y:.2f})",
        xy=(mid_x, mid_y),
        xytext=(mid_x, mid_y + 1000),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    plt.show()


def parse_args():
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Train a linear regression model using the provided dataset and parameters.",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to the input CSV file containing training data. Defaults to 'data/data/data.csv'.",
        default="data/data.csv",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to the output JSON file where the trained parameters (thetas) will be saved. Defaults to 'thetas.json'.",
        default="thetas.json",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for model training. A smaller value indicates slower learning. Defaults to 0.001.",
        default=0.001,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Total number of training iterations (epochs). A higher number can lead to better fitting. Defaults to 5000.",
        default=5_000,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to enable plotting of the training data and regression line after training.",
    )

    parser.add_argument(
        "--error-function",
        type=str,
        help="Select the error function to minimize during training. Options include: "
        + ", ".join(ERROR_FUNCTIONS)
        + ". Defaults to 'MAE'.",
        choices=list(ERROR_FUNCTIONS),
        default="MAE",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode to print detailed error messages during training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        signal.signal(
            signal.SIGINT,
            lambda *_: (print("\033[2Ddslr: CTRL+C sent by user."), exit(1)),
        )

        data = pd.read_csv(args.input_file).dropna()

        validate_csv_structure(data)

        theta1, theta0 = train(
            data,
            args.learning_rate,
            args.epochs,
            args.error_function,
            args.verbose,
        )

        if args.plot:
            plot_result(data, theta1, theta0)

        with open(args.output_file, "w") as thetas_file:
            json.dump(
                {"theta1": theta1, "theta0": theta0},
                thetas_file,
                indent=4,
            )

    except FileNotFoundError:
        print(
            f"Error: The input file '{args.input_file}' was not found.",
            file=sys.stderr,
        )
    except CSVValidationError as ex:
        print(f"{ex.__class__.__name__}: {ex}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
