import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss


def calculate(
    x: np.array, L: int, mode: str, percent_kernel: None | float = None
) -> tuple[np.array, int]:
    """
    Calculates PRSA on based on point 2 from https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param x: values of signal in form of np array vector
    :param L: signal window, where 2L should be larger than the slowest oscillation
    :param mode: "DC" or "AC" (deceleration or acceleration)
    :param percent_kernel: Maximum percentage of change between two values
    :return: prsa values, number of windows
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
    n_windows = len(anchor_points)

    # calculates averages of anchor points for each 'k' index
    X_k = []
    for k in range(-L, L + 1):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k), n_windows


def capacity_capman(prsa_output: np.array) -> float:
    """
    Calculates ascending|descending (AC|DC) capacity of prsa output signal calculated for window L=3,
    like in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2886688/
        The formula for capacity is given by:
        AC|DC = (RR[0] + RR[1] - RR[-1] - RR[-2]) / 4

    :param prsa_output: calculated prsa values for each k
    :return: capacity
    """
    capacity = (prsa_output[0] + prsa_output[1] - prsa_output[-1] - prsa_output[-2]) / 4
    return capacity


def capacity_bauman(prsa_output: np.array, L: int, s: int) -> float:
    """
    Calculates ascending|descending (AC|DC) capacity of prsa output signal calculated for any window L
        The formula for capacity is given by:
        AC|DC = (1 / (2s)) * Σ(x_AC[i]) from i = L+1 to L+s - (1 / (2s)) * Σ(x_AC[i]) from i = L-s+1 to L,
        where:
            - x_AC[i] is the phase-rectified signal for acceleration capacity at point i,
            - L is the anchor point around which the window is considered,
            - s is the parameter for summarizing the phase-rectified curves (assumed to be even).

    :param prsa_output: The input phase-rectified signal.
    :param L: The anchor point around which the AC is calculated.
    :param s: The summarizing parameter for the phase-rectified curves.
    :return: The calculated AC value.
    """
    hl = L // 2
    sum_ac_upper = sum(prsa_output[hl : hl + s])
    sum_ac_lower = sum(prsa_output[hl - s : hl])
    capacity = (1 / (2 * s)) * (sum_ac_upper - sum_ac_lower)

    return capacity


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


def compare_capacities(
    title: str,
    norm_dc: list[float | int],
    norm_ac: list[float | int],
    hyp_dc: list[float | int],
    hyp_ac: list[float | int],
):
    """
    Prints table with comparison of DC and AC capacity for normo- and hypercapnia
    :param title: title of table
    :param norm_dc: list of DC values for normocapnia
    :param norm_ac: list of AC values for normocapnia
    :param hyp_dc: list of DC values for hypercapnia
    :param hyp_ac: list of AC values for hypercapnia
    """

    norm_dc_mean = np.mean(norm_dc)
    norm_dc_std = np.std(norm_dc)
    norm_ac_mean = np.mean(norm_ac)
    norm_ac_std = np.std(norm_ac)
    hip_dc_mean = np.mean(hyp_dc)
    hip_dc_std = np.std(hyp_dc)
    hip_ac_mean = np.mean(hyp_ac)
    hip_ac_std = np.std(hyp_ac)

    result_table = pd.DataFrame(
        [
            [
                f"{norm_dc_mean:.3f} ± {norm_dc_std:.3f}",
                f"{norm_ac_mean:.3f} ± {norm_ac_std:.3f}",
            ],
            [
                f"{hip_dc_mean:.3f} ± {hip_dc_std:.3f}",
                f"{hip_ac_mean:.3f} ± {hip_ac_std:.3f}",
            ],
        ],
        index=["normocapnia", "hypercapnia"],
        columns=["DC", "AC"],
    )

    print(title)
    print(result_table)
