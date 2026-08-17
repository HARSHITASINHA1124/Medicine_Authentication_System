import pandas as pd

try:
    from .config import FEATURES
except ImportError:  # pragma: no cover - direct script execution
    from config import FEATURES


def calibrate(df, calibration_values=None):

    result = df.copy()

    # No calibration values supplied.
    # Return data unchanged for now.
    if calibration_values is None:
        print(
            "Calibration: no calibration reference supplied. "
            "Using raw readings."
        )

        return result

    # ----------------------------------
    # Dark-reference subtraction
    # ----------------------------------

    for feature in FEATURES:

        if feature not in calibration_values:

            raise ValueError(
                f"Missing calibration value for {feature}"
            )

        result[feature] = (
            result[feature]
            - calibration_values[feature]
        )

    print("Calibration applied.")

    return result