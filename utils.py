import pandas as pd


class CSVValidationError(Exception):
    pass


class ColumnMismatchError(CSVValidationError):
    pass


class DtypeMismatchError(CSVValidationError):
    pass


def validate_csv_structure(df):
    expected_cols = ["km", "price"]

    if list(df.columns) != expected_cols:
        raise ColumnMismatchError("Column names do not match the expected structure.")

    for col in expected_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DtypeMismatchError(f"Column '{col}' is not numeric.")

    return True
