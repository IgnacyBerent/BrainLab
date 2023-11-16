import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss


def calculate(
    x: np.array, L: int, mode: str, percent_kernel: None | int = None
) -> np.array:
    """
    Calculates PRSA on based on point 2 from https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param x: values of signal in form of np array vector
    :param L: signal window, where 2L should be larger than the slowest oscillation
    :param mode: "DC" or "AC" (deceleration or acceleration)
    :param percent_kernel: maximum percentage value of change between two values
    :return: prsa values
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
        percent_kernel = percent_kernel / 100
        if mode == "DC":
            # It chooses anchor points which meet condition: X_i < X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] < x[i - 1]
                if abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [
                i
                for i in indexs[L:-L]
                if x[i] > x[i - 1]
                if abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel
            ]

    anchor_points = np.array(anchor_list)

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k)


def get_rr_intervals(signal_df: pd.DataFrame, height: float | int, distance: int) -> np.array:
    """
    Takes signal DataFrame where columns are: 'Values' and 'TimeSteps' and calculates rr intervals
    :param signal_df: dataframe with signal values and time steps
    :param height: minimum height above which will be registered peak
    :param distance: minimum distance between peaks
    :return: rr intervals
    """
    if height > np.max(signal_df["Values"]):
        raise ValueError("Height is too large")

    # finds peaks indexes
    peaks_indexs, _ = ss.find_peaks(x = signal_df["Values"],height = height, distance = distance)
    # makes array of time at wich peaks occured
    peaks_time = signal_df["TimeSteps"].iloc[peaks_indexs].to_numpy() * 0.005
    # calculates intervals between peaks
    rr_intervals = np.diff(peaks_time)
    return np.array(rr_intervals)


def capacity(prsa_values: np.array) -> float:
    """
    Calculates ascending/descending capacity of prsa signal,
    like in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2886688/
    :param prsa_values: calculated prsa values for each k
    :return: capacity
    """
    return (prsa_values[0] + prsa_values[1] - prsa_values[-1] - prsa_values[-2]) / 4


def plot(prsa_values: np.array):
    """
    Plots prsa signal
    :param prsa_values: calculated prsa values for each k
    :return: prsa plot
    """
    x = np.arange(start=-len(prsa_values) // 2, stop=len(prsa_values) // 2)
    y = prsa_values
    y_min = np.min(y) * 0.9
    y_max = np.max(y) * 1.1
    plt.ylim(y_min, y_max)
    plt.scatter(x, y)
    plt.title("Phase Rectified Signal Average")
    plt.xlabel("Index k")
    plt.ylabel("X(k) [mm[Hg]]")
    plt.show()


def plot_rr(prsa_values: np.array):
    """
    Plots prsa signal
    :param prsa_values: calculated prsa values for each k
    :return: prsa plot
    """
    x = np.arange(start=-len(prsa_values) // 2, stop=len(prsa_values) // 2)
    y = prsa_values
    y_min = np.min(y) * 0.9
    y_max = np.max(y) * 1.1
    plt.ylim(y_min, y_max)
    plt.scatter(x, y)
    plt.title("Phase Rectified Signal Average")
    plt.xlabel("Interval number")
    plt.ylabel("RR interval (ms)")
    plt.show()


def calculate_rr_dc_ac(signal_df: pd.DataFrame, percentile: float, distance: float) -> tuple[float, float]:
    """
    Calculates DC and AC capacity for given signal dataframe,
    where columns are: 'Values' and 'TimeSteps'.
    It firslty lineary interpolates signal,
    then calculates rr intervals and finally calculates DC and AC capacity.
    :param signal_df: dataframe with signal values and time steps
    :param percentile: value between 0 and 1, which determines level of cut off
    :param distance: minimum distance between peaks
    :return: DC and AC capacity
    """

    if percentile < 0 or percentile >= 1:
        raise ValueError("Percentile has to be between 0 and 1")

    signal_df.interpolate(inplace=True)
    cut_off = np.max(signal_df["Values"]) * percentile
    rr_signal = get_rr_intervals(signal_df, cut_off, distance)

    prsa__dc = calculate(rr_signal, 3, "DC")
    prsa_ac = calculate(rr_signal, 3, "AC")

    capacity_dc = capacity(prsa__dc)
    capacity_ac = capacity(prsa_ac)

    return capacity_dc, capacity_ac


def compare_capacities(
    title: str,
    norm_dc: list[float],
    norm_ac: list[float],
    hip_dc: list[float],
    hip_ac: list[float],
):
    """
    Prints table with comparison of DC and AC capacity for normo- and hypercapnia
    :param title: title of table
    :param norm_dc: list of DC capacity values for normocapnia
    :param norm_ac: list of AC capacity values for normocapnia
    :param hip_dc: list of DC capacity values for hypercapnia
    :param hip_ac: list of AC capacity values for hypercapnia
    """

    norm_dc_mean = np.mean(norm_dc)
    norm_dc_std = np.std(norm_dc)
    norm_ac_mean = np.mean(norm_ac)
    norm_ac_std = np.std(norm_ac)
    hip_dc_mean = np.mean(hip_dc)
    hip_dc_std = np.std(hip_dc)
    hip_ac_mean = np.mean(hip_ac)
    hip_ac_std = np.std(hip_ac)

    result_table = pd.DataFrame(
        [
            [
                f"{norm_dc_mean:.2f} ± {norm_dc_std:.2f}",
                f"{norm_ac_mean:.2f} ± {norm_ac_std:.2f}",
            ],
            [
                f"{hip_dc_mean:.2f} ± {hip_dc_std:.2f}",
                f"{hip_ac_mean:.2f} ± {hip_ac_std:.2f}",
            ],
        ],
        index=["normocapnia", "hypercapnia"],
        columns=["DC", "AC"],
    )

    print(title)
    print(result_table)
