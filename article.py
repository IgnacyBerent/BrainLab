import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss

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


def get_rr_intervals_with_time(signal_df: pd.DataFrame, height: float | int, distance: int, show_plot: bool, plot_title) -> dict:
    """
    Takes signal DataFrame where columns are: 'Values' and 'TimeSteps' and calculates rr intervals
    :param signal_df: dataframe with signal values and time steps
    :param height: minimum height above which will be registered peak
    :param distance: minimum distance between peaks
    :param show_plot: if True, it shows plot with peaks
    :return: dictionary with time of occurence of peak and its rr-value
    """
    if height > np.max(signal_df["Values"]):
        raise ValueError("Height is too large")

    # finds peaks indexes
    peaks_indexs, _ = ss.find_peaks(x = signal_df["Values"],height = height, distance = distance)
    # makes array of time at wich peaks occured
    peaks_time = signal_df["TimeSteps"].iloc[peaks_indexs].to_numpy() * 0.005
    # calculates intervals between peaks
    rr_intervals = np.diff(peaks_time)
    # make dictionary with time of occurence of peak and its rr-value
    rr_intervals_dict = dict(zip(peaks_time[1:], rr_intervals))

    if show_plot:
        signal_df.plot.scatter(x='TimeSteps', y='Values')
        plt.title(plot_title)
        plt.scatter(signal_df['TimeSteps'][peaks_indexs], signal_df['Values'][peaks_indexs], c='r')

    return rr_intervals_dict


def plot_with_anchors(rr_signal_dict: dict, anchor_points: np.array, L: int):
    """
    Makes Scatter plot of signal with anchor points
    :param signal: signal to plot
    :param anchor_points: anchor points
    :param L: window size
    :return: None
    """
    # make scatter plot of signal with connected points
    # make the middle level at 1.0, and let y axis be called
    # RR Interval (s)
    # X axis are keys of rr_signal_dict
    # Y axis are values of rr_signal_dict
    # make x axis be called Time (s)
    # anchor points are indexes of anchor points on Time axis of value
    # eqyal to signal at that point
    # make anchor points be red and have shape of x

    plt.scatter(rr_signal_dict.keys(), rr_signal_dict.values(), c='b')
    plt.scatter(list(rr_signal_dict.keys())[anchor_points], list(rr_signal_dict.values())[anchor_points], c='r', marker='x')
    plt.title("RR intervals with anchor points")
    plt.xlabel("Time (s)")
    plt.ylabel("RR Interval (s)")
    plt.show()
    
    


