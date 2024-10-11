import sys
import json
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from predict import estimate_price

ERROR_FUNCTIONS = {"MAE": abs, "MSE": lambda x: x * x}


def normalize(value, mean, std):
    return (value - mean) / std


def mean(values):
    return sum(values) / len(values)


def variance(values, mean):
    return sum((x - mean) ** 2 for x in values) / len(values)


def standard_deviation(values, mean):
    return math.sqrt(variance(values, mean))


def train(data, l_rate, epochs, error_func, threshold):
    theta1 = theta0 = 0
    mileages = data["km"]
    prices = data["price"]
    prices_mean = mean(prices)
    mileages_mean = mean(mileages)
    prices_std = standard_deviation(prices, prices_mean)
    mileages_std = standard_deviation(mileages, mileages_mean)
    prices = [normalize(p, prices_mean, prices_std) for p in prices]
    mileages = [normalize(m, mileages_mean, mileages_std) for m in mileages]
    m = len(data)
    for _ in tqdm(range(epochs)):
        tmp0 = l_rate * (
            sum(
                estimate_price(theta1, theta0, mileages[i]) - prices[i]
                for i in range(m)
            )
            / m
        )
        tmp1 = l_rate * (
            sum(
                (estimate_price(theta1, theta0, mileages[i]) - prices[i]) * mileages[i]
                for i in range(m)
            )
            / m
        )
        theta1 -= tmp1
        theta0 -= tmp0

        C = theta1 * (prices_std / mileages_std)
        error = mean(
            [
                error_func(
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

    theta1 = theta1 * (prices_std / mileages_std)
    theta0 = prices_mean - theta1 * mileages_mean
    error = mean(
        [
            error_func(estimate_price(theta1, theta0, km) - price)
            for price, km in zip(data["price"], data["km"])
        ]
    )
    print(error)
    return theta1, theta0


def plot_result(data, theta1, theta0):
    y_line = [(theta1 * x + theta0) for x in data["km"]]
    plt.figure("Linear regression results", figsize=(10, 5))
    plt.scatter(data["km"], data["price"], color="blue", alpha=0.7)
    plt.plot(data["km"], y_line, color="red", label=f"y = {theta1}x + {theta0}")
    plt.xlabel("Kilometers Driven (km)")
    plt.ylabel("Price")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Train the model on given datas.",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Path of the input csv file with training data.",
        default="data/data.csv",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path of the output json file with trained thetas.",
        default="thetas.json",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate to use for training.",
        default=0.001,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training iterations.",
        default=5_000,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the function and data after training.",
    )

    parser.add_argument(
        "--error-function",
        type=str,
        help="Choose the error function. Defaults to MAE (average of the absolute differences between predicted values and actual values).",
        choices=list(ERROR_FUNCTIONS),
        default="MAE",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Choose the threshold percentage. Exemple: 0.1 will equal 10%",
    )

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_file)
        assert "km" in data and "price" in data, "Invalid input data."
        error_func = ERROR_FUNCTIONS[args.error_function]
        theta1, theta0 = train(
            data, args.learning_rate, args.epochs, error_func, args.threshold
        )
        if args.plot:
            plot_result(data, theta1, theta0)
        with open(args.output_file, "w") as thetas:
            json.dump(
                {"theta1": theta1, "theta0": theta0},
                thetas,
                indent=4,
            )
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
