import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw Spaceship Titanic data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features.
    """
    df = df.copy()

    # =========================
    # Cabin-derived features
    # =========================
    df["Deck"] = df["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) and "/" in str(x) else "Unknown"
    )
    df["CabinNum"] = df["Cabin"].apply(
        lambda x: x.split("/")[1] if pd.notna(x) and "/" in str(x) else None
    )
    df["Side"] = df["Cabin"].apply(
        lambda x: x.split("/")[2] if pd.notna(x) and "/" in str(x) else "Unknown"
    )

    # =========================
    # PassengerId-derived features
    # =========================
    df["GroupID"] = df["PassengerId"].apply(
        lambda x: x.split("_")[0] if pd.notna(x) and "_" in str(x) else "Unknown"
    )
    df["PassengerNumber"] = df["PassengerId"].apply(
        lambda x: x.split("_")[1] if pd.notna(x) and "_" in str(x) else None
    )

    df["GroupSize"] = df.groupby("GroupID")["GroupID"].transform("count")
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)

    # =========================
    # Name-derived features
    # =========================
    df["LastName"] = df["Name"].apply(
        lambda x: x.split()[-1] if pd.notna(x) and len(str(x).split()) > 0 else "Unknown"
    )
    df["FamilySize"] = df.groupby("LastName")["LastName"].transform("count")

    # =========================
    # Spending features
    # =========================
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[spending_cols].fillna(0).sum(axis=1)
    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)

    # =========================
    # Age group
    # =========================
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 50, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    ).astype("object")

    # =========================
    # Missing indicators
    # =========================
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["CryoSleepMissing"] = df["CryoSleep"].isna().astype(int)
    df["VIPMissing"] = df["VIP"].isna().astype(int)

    # =========================
    # Numeric conversion
    # =========================
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
    df["PassengerNumber"] = pd.to_numeric(df["PassengerNumber"], errors="coerce")

    return df