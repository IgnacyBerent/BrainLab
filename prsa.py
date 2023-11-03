import numpy as np
import matplotlib.pyplot as plt


def calculate(x: np.array, L: int, mode: str, percent_kernel: None | int = None) -> np.array:
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
            anchor_list = [i for i in indexs[L:-L] if x[i] < x[i - 1] if abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel]
        elif mode == "AC":
            # It chooses anchor points which meet condition: X_i > X_i-1,
            anchor_list = [i for i in indexs[L:-L] if x[i] > x[i - 1] if abs(x[i] - x[i - 1]) / x[i - 1] < percent_kernel]


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

def plot_rr(prsa_values: np.array):
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
    plt.xlabel("Interval number")
    plt.ylabel("RR interval (ms)")
    plt.show()

def capacity(prsa_values: np.array) -> float:
    """
    Calculates ascending/descending capacity of prsa signal,
    like in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2886688/
    :param prsa_values: calculated prsa values for each k
    :return: capacity
    """
    return (prsa_values[0] + prsa_values[1] - prsa_values[-1] - prsa_values[-2]) / 4
