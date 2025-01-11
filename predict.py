from argparse import ArgumentParser
import json
import signal
import sys


def estimate_price(theta1, theta0, mileage):
    return theta1 * mileage + theta0


def parse_args():
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Predict a price based on training results.",
    )

    parser.add_argument(
        "--file",
        type=str,
        default="thetas.json",
        help="Path to the JSON file containing trained thetas. Defaults to 'thetas.json' if not specified.",
    )

    parser.add_argument(
        "--mileage",
        type=float,
        help="The mileage of the vehicle in miles. This value is used to predict the price based on mileage.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        signal.signal(
            signal.SIGINT,
            lambda *_: (print("\033[2Ddslr: CTRL+C sent by user."), exit(1)),
        )

        with open(args.file, "r") as thetas:
            data = json.load(thetas)
            theta1, theta0 = data["theta1"], data["theta0"]

        mileage = args.mileage if args.mileage is not None else float(input("Enter your mileage: "))

        print(estimate_price(theta1, theta0, mileage))

    except FileNotFoundError:
        print(f"Error: The file '{args.file}' was not found.", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Error: The file '{args.file}' contains invalid JSON.", file=sys.stderr)
    except KeyError as e:
        print(f"Error: Missing expected key in JSON data: {e}", file=sys.stderr)
    except ValueError:
        print(
            "Error: Invalid mileage input. Please enter a numeric value.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
