import pandas as pd


def apply_training_categories(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list[str]],
]:
    """Apply category levels learned only from the training split."""
    train = X_train.copy()
    validation = X_validation.copy()
    test = X_test.copy()
    categorical_levels: dict[
        str,
        list[str],
    ] = {}

    for column in categorical_features:
        levels = sorted(
            train[column]
            .dropna()
            .astype("string")
            .unique()
            .tolist()
        )

        if not levels:
            raise ValueError(
                "Categorical training feature has no "
                f"observed values: {column}"
            )

        categorical_levels[column] = levels
        category_type = pd.CategoricalDtype(
            categories=levels,
            ordered=False,
        )

        for frame in (
            train,
            validation,
            test,
        ):
            frame[column] = (
                frame[column]
                .astype("string")
                .astype(category_type)
            )

    return (
        train,
        validation,
        test,
        categorical_levels,
    )


def apply_saved_categories(
    features: pd.DataFrame,
    categorical_levels: dict[
        str,
        list[str],
    ],
) -> pd.DataFrame:
    """Apply category levels stored with a trained model."""
    result = features.copy()

    for column, levels in categorical_levels.items():
        if column not in result.columns:
            raise ValueError(
                "Inference data is missing categorical "
                f"feature: {column}"
            )

        category_type = pd.CategoricalDtype(
            categories=levels,
            ordered=False,
        )
        result[column] = (
            result[column]
            .astype("string")
            .astype(category_type)
        )

    return result
