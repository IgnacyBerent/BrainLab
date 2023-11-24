import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate(
    x: np.array, L: int, mode: str, percent_kernel: None | float = None
) -> tuple[np.array, np.array]:
    """
    Calculates PRSA on based on point 2 from https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param x: values of signal in form of np array vector
    :param L: signal window, where 2L should be larger than the slowest oscillation
    :param mode: "DC" or "AC" (deceleration or acceleration)
    :param percent_kernel: Maximum percentage of change between two values
    :return: prsa values, anchor points indexes
    """

    if mode != "DC" and mode != "AC":
        raise ValueError('Mode has to be "DC" or "AC"')
    if len(x) < 2 * L:
        raise ValueError("Signal window is too large")
    if percent_kernel < 0:
        raise ValueError("Kernel can't be negative")

    indexs = np.arange(len(x))

    # It doesn't take under consideration first and last L points
    if percent_kernel is None:
        if mode == "DC":
            # It chooses anchor points which meet condition: X_i < X_i-1,
            anchor_list = [i for i in indexs[L:-L] if x[i] < x[i - 1]]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [i for i in indexs[L:-L] if x[i] > x[i - 1]]
    else:
        if mode == "DC":
            # It chooses anchor points which meet condition: X_i < X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] < x[i - 1]
                and abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] > x[i - 1]
                and abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]

    anchor_points = np.array(anchor_list)
    n_windows = len(anchor_points)

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k), anchor_points


def plot_with_anchors(signal: np.array, anchor_points: np.array, L: int):
    """
    Plots signal with anchor points
    :param signal: signal to plot
    :param anchor_points: anchor points
    :param L: window size
    :return: None
    """
    plt.plot(signal)
    plt.plot(anchor_points, signal[anchor_points], "o")
    plt.plot(
        anchor_points + L,
        signal[anchor_points + L],
        "o",
        color="red",
        label="anchor points",
    )
    plt.plot(
        anchor_points - L,
        signal[anchor_points - L],
        "o",
        color="red",
    )
    plt.legend()
    plt.show()

