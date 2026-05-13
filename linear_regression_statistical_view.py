import numpy as np
from scipy import stats


class LinearRegressionStatisticalView:

    def __init__(self, alpha: float = 0.05):

        self.alpha = alpha
        self.fitted = False
        self.B_0 = None
        self.B_1 = None
        self.n = None
        self.x_bar = None
        self.y_bar = None
        self.SXX = None
        self.SSE = None
        self.SST = None
        self.SSR = None
        self.df_model = None
        self.df_error = None
        self.df_total = None


    def fit(self, X, Y):

        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)

        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have same length")

        self.n = len(self.X)

        self.x_mean = np.mean(self.X)
        self.y_mean = np.mean(self.Y)

        self.beta_1 = (
            np.sum((self.X - self.x_mean) * (self.Y - self.y_mean))
            /
            np.sum((self.X - self.x_mean) ** 2)
        )

        self.beta_0 = self.y_mean - self.beta_1 * self.x_mean

        self.y_hat = self.beta_0 + self.beta_1 * self.X

        self.residuals = self.Y - self.y_hat

        self.SST = np.sum((self.Y - self.y_mean) ** 2)

        self.SSR = np.sum((self.y_hat - self.y_mean) ** 2)

        self.SSE = np.sum((self.Y - self.y_hat) ** 2)

        self.df_model = 1
        self.df_resid = self.n - 2
        self.df_total = self.n - 1

        self.MSR = self.SSR / self.df_model

        self.MSE = self.SSE / self.df_resid

        self.F = self.MSR / self.MSE

        self.R2 = self.SSR / self.SST

        self.R = np.sign(self.beta_1) * np.sqrt(self.R2)

        self.fitted = True

    def f_statistic(self) -> float:

        return float(self.F)

    def f_critical(self) -> float:

        return float(
            stats.f.ppf(
                1 - self.alpha,
                self.df_model,
                self.df_resid
            )
        )

    def beta_confidence_intervals(self) -> dict:

        Sxx = np.sum((self.X - self.x_mean) ** 2)

        se_beta_1 = np.sqrt(self.MSE / Sxx)

        se_beta_0 = np.sqrt(
            self.MSE * (
                (1 / self.n)
                +
                (self.x_mean ** 2 / Sxx)
            )
        )

        t_critical = stats.t.ppf(
            1 - self.alpha / 2,
            self.df_resid
        )

        beta_0_interval = (
            self.beta_0 - t_critical * se_beta_0,
            self.beta_0 + t_critical * se_beta_0
        )

        beta_1_interval = (
            self.beta_1 - t_critical * se_beta_1,
            self.beta_1 + t_critical * se_beta_1
        )

        return {
            "beta_0": beta_0_interval,
            "beta_1": beta_1_interval
        }

    def degrees_of_freedom(self) -> dict:

        return {
            "df_model": self.df_model,
            "df_resid": self.df_resid,
            "df_total": self.df_total
        }

    def r_squared(self) -> float:

        return float(self.R2)

    def r(self) -> float:

        return float(self.R)

    def summary(self) -> dict:

        return {
            "f_critical": self.f_critical(),
            "f_stat": self.f_statistic(),
            "beta_confidence_intervals":
                self.beta_confidence_intervals(),
            "degrees_of_freedom":
                self.degrees_of_freedom(),
            "r_squareddef": self.r_squared(),
            "r": self.r()
        }


def compare(
    df,
    x_col_1: str,
    x_col_2: str,
    y_col: str,
    alpha: float = 0.05
) -> dict:

    model_1 = LinearRegressionStatisticalView(alpha)

    model_2 = LinearRegressionStatisticalView(alpha)

    model_1.fit(
        df[x_col_1],
        df[y_col]
    )

    model_2.fit(
        df[x_col_2],
        df[y_col]
    )

    f_stat_1 = model_1.f_statistic()

    f_stat_2 = model_2.f_statistic()

    if f_stat_1 > f_stat_2:
        best_predictor = x_col_1
    else:
        best_predictor = x_col_2

    return {
        "predictor_1": x_col_1,
        "predictor_2": x_col_2,
        "f_stat_1": f_stat_1,
        "f_stat_2": f_stat_2,
        "best_predictor": best_predictor
    }



