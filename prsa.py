import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate(df: pd.DataFrame, L: int) -> np.array:
    """
    Calculates PRSA on based on point 2 from https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param df: signal data with 'abp[mmHg]' column
    :param L: signal window, where 2L has to be larger than slowest oscillation
    :return: prsa values
    """

    x = df['abp[mmHg]'].to_numpy()
    indexs = np.arange(len(x))

    # It chooses anchor points which meet condition: X_i > X_i-1,
    # It doesn't take under consideration first and last L points
    anchor_list = [i for i in indexs[L:-L] if x[i] > x[i - 1]]
    anchor_points = np.array(anchor_list)

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k)


def plot(prsa_values: np.array):
    """
    Plots prsa signal
    :param prsa_values: calculated prsa values for each k
    :return: prsa plot
    """
    x = np.arange(start=-len(prsa_values)//2, stop=len(prsa_values)//2)
    y = prsa_values
    y_min = np.min(y) * 0.9
    y_max = np.max(y) * 1.1
    plt.ylim(y_min, y_max)
    plt.scatter(x, y)
    plt.title("Phase Rectified Signal Average")
    plt.xlabel("Index k")
    plt.ylabel("X(k) [mm[Hg]]")
    plt.show()
