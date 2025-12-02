import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-v0_8")


def load_health_data(path: str = "data/health_study_dataset.csv", seed: int = 42) -> pd.DataFrame:
    """
    Läser in hälsodatat från en CSV-fil och sätter ett slumpseed.

    Parametrar
    ----------
    path : str
       Sökväg till CSV-filen.
    seed : int
       Slumpseed för numpy (så att simuleringar blir repeterbara).
    
    Returnerar
    ----------
    pd.DataFrame
       En DataFrame med hälsodatat.
    """

    np.random.seed(seed)
    df = pd.read_csv(path)
    return df


def calc_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar medel, median, min och max för viktiga kolumner.

    Parametrar
    ----------
    df : pd.DataFrame
        Hälsodatat.

    Returnerar
    ----------
    pd.DataFrame
        En tabell med beskrivande statistik.
    """
    cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
    return df[cols].agg(["mean", "median", "min", "max"]).round(2)


def plot_bp_hist(df: pd.DataFrame) -> None:
    """
    Ritar ett histogram över systoliskt blodtryck.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df["systolic_bp"], bins=20)
    plt.xlabel("Systoliskt blodtryck (mmHg)")
    plt.ylabel("Antal personer")
    plt.title("Histogram över systoliskt blodtryck")
    plt.show()


def plot_bp_vs_age(df: pd.DataFrame) -> None:
    """
    Ritar ett spridningsdiagram (scatterplot) mellan ålder och systoliskt blodtryck.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df["age"], df["systolic_bp"], alpha=0.6)
    plt.xlabel("Ålder (år)")
    plt.ylabel("Systoliskt blodtryck (mmHg)")
    plt.title("Relation mellan ålder och systoliskt blodtryck")
    plt.show()


class HealthAnalyzer:
    """
    Klass som kan göra olika analyser på hälsodatat.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Skapar en HealthAnalyzer.

        Parametrar
        ----------
        df : pd.DataFrame
            Hälsodatat som ska analyseras.
        """
        self.df = df
        self.bp_model = None

    def summary_stats(self) -> pd.DataFrame:
        """
        Returnerar enkel beskrivande statistik för några centrala variabler.
        """
        return calc_summary_stats(self.df)

    def plot_bp_hist(self) -> None:
        """
        Ritar ett histogram över systoliskt blodtryck.
        """
        plot_bp_hist(self.df)

    def plot_bp_vs_age(self) -> None:
        """
        Ritar en scatterplot mellan ålder och systoliskt blodtryck.
        """
        plot_bp_vs_age(self.df)

    def bp_regression(self):
        """
        Gör en enkel linjär regression där systoliskt blodtryck förklaras av ålder och vikt.

        Returnerar
        ----------
        model : LinearRegression
            Den tränade regressionsmodellen.
        r2 : float
            Förklaringsgrad (R^2) för modellen.
        """
        X = self.df[["age", "weight"]]
        y = self.df["systolic_bp"]

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)
        self.bp_model = model

        return model, r2
