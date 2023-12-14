import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss


def calculate(
        x: np.array, L: int, mode: str, percent_kernel: None | float = None
) -> tuple[np.array, np.array, np.array]:
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
        if percent_kernel < 0:
            raise ValueError("Kernel can't be negative")
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

    # for each anchor point take values from window of size 2L
    anchor_neighbours = []
    for i in anchor_points:
        anchor_neighbours.append(x[i - L: i + L + 1] * 1000)

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L + 1):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k), anchor_points, anchor_neighbours


def get_rr_intervals_with_time(
        signal_df: pd.DataFrame,
        height: float | int,
        distance: int,
        show_plot: bool, plot_title: str, y_label: str
) -> dict:
    """
    Takes signal DataFrame where columns are: 'Values' and 'TimeSteps' and calculates rr intervals
    :param signal_df: dataframe with signal values and time steps
    :param height: minimum height above which will be registered peak
    :param distance: minimum distance between peaks
    :param show_plot: if True, it shows plot with peaks
    :param plot_title: title of plot
    :param y_label: label of y axis
    :return: dictionary with time of occurence of peak and its rr-value
    """
    if height > np.max(signal_df["Values"]):
        raise ValueError("Height is too large")

    # finds peaks indexes
    peaks_indexs, _ = ss.find_peaks(x=signal_df["Values"], height=height, distance=distance)
    # makes array of time at wich peaks occured
    peaks_time = signal_df["TimeSteps"].iloc[peaks_indexs].to_numpy() * 0.005
    # calculates intervals between peaks
    rr_intervals = np.diff(peaks_time)
    # make dictionary with time of occurence of peak and its rr-value
    rr_intervals_dict = dict(zip(peaks_time[1:], rr_intervals))

    if show_plot:
        signal_df.plot.scatter(x='TimeSteps', y='Values')
        plt.title(plot_title)
        plt.scatter(signal_df['TimeSteps'][peaks_indexs], signal_df['Values'][peaks_indexs], c='r', s=15)
        plt.legend(['Sygnał', 'Szczyty'])
        plt.ylabel(y_label)
        plt.show()

    return rr_intervals_dict


def plot_with_anchors(
        rr_signal_dict: dict,
        ac_anchor_points: np.array,
        dc_anchor_points: np.array,
        L: int
) -> None:
    """
    Makes Scatter plot of signal with anchor points
    :param rr_signal_dict: signal to plot
    :param ac_anchor_points: accelerating anchor points
    :param dc_anchor_points: decelerating anchor points
    :param L: window size
    :return: None
    """
    keys_list = list(rr_signal_dict.keys())
    values_list = list(rr_signal_dict.values())
    keys_list = keys_list[L:-L]
    values_list = values_list[L:-L]

    # adjust anchor points indexises to shortened signal
    ac_anchor_points = ac_anchor_points - L
    dc_anchor_points = dc_anchor_points - L

    # Plot line connecting points
    plt.plot(keys_list, values_list, c='b')
    plt.scatter([keys_list[i] for i in ac_anchor_points], [values_list[i] for i in ac_anchor_points], c='r', marker='x')
    plt.scatter([keys_list[i] for i in dc_anchor_points], [values_list[i] for i in dc_anchor_points], c='g', marker='o')

    # add legend that describes anchor points
    plt.legend(['RR interwały', 'AC punkty', 'DC punkty'])

    plt.title("RR interwały z sąsiadami")
    plt.ylabel("RR Interwał (s)")
    plt.show()


def plot_all_anchors_with_neighbours(anchor_neighbours: np.array, L: int) -> None:
    """
    Plots all anchor points with their neighbours
    :param anchor_neighbours: array of anchor points with their neighbours
    :param L: window size
    :return: None
    """
    for i in range(len(anchor_neighbours)):
        plt.plot(np.arange(-L, L + 1), anchor_neighbours[i])
    plt.xlabel("Numer indeksu (w zależności od kotwicy)")
    plt.ylabel("RR Interwał (ms)")
    plt.title('Wszystkie punkty kotwicze z ich sąsiadami')
    plt.show()
