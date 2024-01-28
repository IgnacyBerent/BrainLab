import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import peaks_detection


def calculate(
    x: np.array, L: int, mode: str, percent_kernel: None | float = None
) -> tuple[np.array, np.array, np.array]:
    """
    Calculates PRSA on based on point 2 from https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param x: values of signal in form of np array vector
    :param L: signal window, where 2L should be larger than the slowest oscillation
    :param mode: "DC" or "AC" (deceleration or acceleration)
    :param percent_kernel: Maximum percentage of change between two values
    :return: prsa values, anchor points indexes and anchor points with their neighbours
    """

    if mode != "DC" and mode != "AC":
        raise ValueError('Mode has to be "DC" or "AC"')
    if len(x) < 2 * L:
        raise ValueError("Signal window is too large")

    indexs = np.arange(len(x))
    anchor_list = None

    # It doesn't take under consideration first and last L points
    if percent_kernel is None:
        if mode == "DC":
            # It chooses anchor points which meet condition: X_i < X_i-1,
            anchor_list = [i for i in indexs[L:-L] if x[i] < x[i - 1]]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [i for i in indexs[L:-L] if x[i] > x[i - 1]]
    else:
        if percent_kernel < 0:
            raise ValueError("Kernel can't be negative")
        if mode == "DC":
            # It chooses anchor points which meet condition: X_i < X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] < x[i - 1] and abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] > x[i - 1] and abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]

    anchor_points = np.array(anchor_list)

    # for each anchor point take values from window of size 2L
    anchor_neighbours = []
    for i in anchor_points:
        anchor_neighbours.append(x[i - L : i + L + 1])

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L + 1):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k), anchor_points, anchor_neighbours


def plot_with_anchors(
    rr: np.array,
    ac_anchors_indexes: np.array,
    dc_anchors_indexes: np.array,
    to_seconds: bool = True,
) -> None:
    """
    Makes Scatter plot of signal with anchor points
    :param rr: rr signal
    :param ac_anchors_indexes: indexes of accelerating anchor points
    :param dc_anchors_indexes: indexes of decelerating anchor points
    :param to_seconds: if True, x-axis is in seconds, otherwise in samples
    """

    # plot rr signal as line and accelerating anchor points as green dots and decelerating as red 'x'
    plt.figure(figsize=(40, 15))

    x_axis = np.cumsum(rr) / 1000 if to_seconds else np.cumsum(rr)
    x_label = "Time (seconds)" if to_seconds else "Time (ms)"

    plt.plot(
        x_axis,
        rr,
        label="RR-interval",
        color="#A651D8",
        linewidth=2,
    )
    plt.scatter(
        x_axis[ac_anchors_indexes],
        rr[ac_anchors_indexes],
        color="green",
        marker=".",
        s=100,
        label="AC anchor points",
    )
    plt.scatter(
        x_axis[dc_anchors_indexes],
        rr[dc_anchors_indexes],
        color="red",
        marker="x",
        s=100,
        label="DC anchor points",
    )

    plt.title("RR-intervals", fontsize=24)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("RR-interval (ms)", fontsize=16)
    plt.legend(fontsize=16)


def plot_all_anchors_with_neighbours(
    anchor_neighbours: np.array, prsa: np.array, anchors_type: str, L: int
) -> None:
    """
    Plots all anchor points with their neighbours
    :param anchor_neighbours: array of anchor points with their neighbours
    :param prsa: array of PRSA values
    :param anchors_type: "AC" or "DC"
    :param L: signal window, where 2L should be larger than the slowest oscillation
    :raises ValueError: if type is not "AC" or "DC"
    """

    if anchors_type == "AC":
        title = "Accelerating anchor points with their neighbours"
    elif anchors_type == "DC":
        title = "Decelerating anchor points with their neighbours"
    else:
        raise ValueError('Type has to be "AC" or "DC"')

    for i in range(len(anchor_neighbours)):
        plt.plot(np.arange(-L, L + 1), anchor_neighbours[i])

    plt.plot(np.arange(-L, L + 1), prsa, label="PRSA", color="black", linewidth=2)

    plt.xlabel("Index (relative to anchor point)")
    plt.ylabel("RR-interval (ms)")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_signal(
    df: pd.DataFrame,
    start: int,
    stop: int,
    sampling_rate: int,
    filtered: bool = False,
):
    y_label = "Amplitude (arbitrary unit)" if filtered else "Amplitude (mmHg)"
    title = "Filtered ABP signal" if filtered else "Unfiltered ABP signal"
    plt.figure(figsize=(40, 10))
    duration = (stop - start) / sampling_rate
    plt.title(title + ", slice of %.1f seconds" % duration, fontsize=24)
    plt.plot(
        df[start:stop]["TimeSteps"],
        df[start:stop]["Values"],
        color="#51A6D8",
        linewidth=1,
    )
    plt.xlabel("TimeSteps", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
