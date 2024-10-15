import sys
import json
from argparse import ArgumentParser

EXIT_FAILURE = 1


def estimate_price(theta1, theta0, mileage):
    return theta1 * mileage + theta0


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Predict a price based on training results.",
    )

    parser.add_argument(
        "--file",
        type=str,
        default="thetas.json",
        help=(
            "Path to the JSON file containing trained thetas. "
            "Defaults to 'thetas.json' if not specified."
        ),
    )

    parser.add_argument(
        "--mileage",
        type=float,
        help="The mileage of the vehicle in miles. This value is used to predict the price based on mileage.",
    )

    args = parser.parse_args()

    try:
        with open(args.file, "r") as thetas:
            data = json.load(thetas)
            theta1, theta0 = data["theta1"], data["theta0"]

        mileage = (
            args.mileage
            if args.mileage is not None
            else float(input("Enter your mileage: "))
        )

        print(estimate_price(theta1, theta0, mileage))

    except FileNotFoundError:
        print(f"Error: The file '{args.file}' was not found.", file=sys.stderr)
        exit(EXIT_FAILURE)
    except json.JSONDecodeError:
        print(f"Error: The file '{args.file}' contains invalid JSON.", file=sys.stderr)
        exit(EXIT_FAILURE)
    except KeyError as e:
        print(f"Error: Missing expected key in JSON data: {e}", file=sys.stderr)
        exit(EXIT_FAILURE)
    except ValueError:
        print(
            "Error: Invalid mileage input. Please enter a numeric value.",
            file=sys.stderr,
        )
        exit(EXIT_FAILURE)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        exit(EXIT_FAILURE)
