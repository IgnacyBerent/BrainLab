"""
Functions based on https://medium.com/orikami-blog/exploring-heart-rate-variability-using-python-483a7037c64d
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import zscore


def rr_intervals(
    signal: np.array,
    sampling_rate: int = 200,
    threshold: float = 0.45,
    filtered: bool = False,
) -> np.array:
    """
    Calculates rr intervals from abp signal
    :param signal: 1 dimensional np array of abp signal
    :param sampling_rate: sampling rate of the signal
    :param threshold: threshold for peak detection from similarity
    :param filtered: if True, it uses default filter on rr interval
    :return: rr intervals
    """
    peaks, similarity = detect_peaks(signal, threshold=threshold)
    grouped_peaks = group_peaks(peaks)
    rr = np.diff(grouped_peaks)
    rr *= 1000 / sampling_rate  # convert to milliseconds

    if not filtered:
        return rr
    else:
        rr_corrected = rr.copy()
        rr_corrected[np.abs(zscore(rr)) > 2] = np.median(rr)
        return rr_corrected


def plot_rr_intervals(rr: np.array, to_seconds: bool = False) -> None:
    """
    Plots rr rr
    :param rr: rr intervals
    :param to_seconds: if True, x-axis is in seconds, otherwise in milliseconds
    """

    x_label = "Time (seconds)" if to_seconds else "Time (ms)"

    plt.figure(figsize=(40, 15))
    plt.title("RR-intervals", fontsize=24)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("RR-interval (ms)", fontsize=16)

    x_axis = np.cumsum(rr) / 1000 if to_seconds else np.cumsum(rr)

    plt.plot(
        x_axis,
        rr,
        label="RR-interval",
        color="#A651D8",
        linewidth=2,
    )


def plot_abp_vs_rr_intervals(
    df: pd.DataFrame,
    sampfrom: int,
    sampto: int,
    nr_plots: int = 1,
    threshold: float = 0.45,
    sampling_rate: int = 200,
    filtered: bool = False,
):
    peaks, similarity = detect_peaks(df["Values"], threshold=threshold)
    grouped_peaks = group_peaks(peaks)
    rr = np.diff(grouped_peaks)

    for start, stop in get_plot_ranges(sampfrom, sampto, nr_plots):
        # plot similarity
        plt.figure(figsize=(40, 20))

        plt.title("ABP signal & RR-intervals", fontsize=24)
        plt.plot(
            df.index / sampling_rate,
            df["Values"],
            label="ABP",
            color="#51A6D8",
            linewidth=1,
        )
        plt.plot(
            grouped_peaks / sampling_rate,
            np.repeat(60, grouped_peaks.shape[0]),
            markersize=14,
            label="Found peaks",
            color="orange",
            marker="o",
            linestyle="None",
        )
        plt.legend(loc="upper left", fontsize=20)
        plt.xlabel("Time (seconds)", fontsize=16)
        plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)
        plt.gca().set_ylim(-30, 80)

        ax2 = plt.gca().twinx()
        ax2.plot(
            (np.cumsum(rr) + peaks[0]) / sampling_rate,
            rr,
            label="RR-intervals",
            color="#A651D8",
            linewidth=2,
            markerfacecolor="#A651D8",
            markeredgewidth=0,
            marker="o",
            markersize=18,
        )
        ax2.set_xlim(start / sampling_rate, stop / sampling_rate)
        ax2.set_ylim(-500, 500)
        ax2.legend(loc="upper right", fontsize=20)

        plt.xlabel("Time (seconds)", fontsize=16)
        plt.ylabel("RR-interval (ms)", fontsize=16)


def plot_abp_with_similarity(
    df: pd.DataFrame,
    sampfrom: int,
    sampto: int,
    nr_plots: int = 1,
    threshold: float = 0.45,
):
    for start, stop in get_plot_ranges(sampfrom, sampto, nr_plots):
        cond_slice = (df.index >= start) & (df.index < stop)
        abp_slice = df["Values"][cond_slice]

        # detect peaks
        peaks, similarity = detect_peaks(abp_slice, threshold=threshold)

        # plot similarity
        plt.figure(figsize=(40, 20))

        plt.subplot(211)
        plt.title("ABP signal with found peaks", fontsize=24)
        plt.plot(abp_slice.index, abp_slice, label="ABP", color="#51A6D8", linewidth=1)
        plt.plot(
            peaks,
            np.repeat(60, peaks.shape[0]),
            markersize=10,
            label="peaks",
            color="orange",
            marker="o",
            linestyle="None",
        )
        plt.legend(loc="upper right", fontsize=20)
        plt.xlabel("TimeSteps", fontsize=16)
        plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)

        plt.subplot(212)
        plt.title("Similarity plot", fontsize=24)
        plt.plot(
            abp_slice.index,
            similarity,
            label="Similarity with filter",
            color="olive",
            linewidth=1,
        )
        plt.plot(threshold * np.ones(len(similarity)), label="Threshold", color="red")
        plt.legend(loc="upper right", fontsize=20)
        plt.xlabel("TimeSteps", fontsize=16)
        plt.ylabel("Similarity (normalized)", fontsize=16)


def get_plot_ranges(start=10, end=20, n=5):
    """
    Make an iterator that divides into n or n+1 ranges.
    - if end-start is divisible by steps, return n ranges
    - if end-start is not divisible by steps, return n+1 ranges, where the last range is smaller and ends at n

    # Example:
    >> list(get_plot_ranges())
    >> [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]

    """
    distance = end - start
    for i in np.arange(start, end, np.floor(distance / n)):
        yield (int(i), int(np.minimum(end, np.floor(distance / n) + i)))


def group_peaks(p, threshold: int = 5) -> np.array:
    """
    The peak detection algorithm finds multiple peaks for each signal wave.
    Here we group collections of peaks that are very near (within threshold) and we take the median index
    :param p: peak indexes
    :param threshold: threshold for grouping peaks
    :return: grouped peaks
    """
    # initialize output
    output = np.empty(0)

    # label groups of sample that belong to the same peak
    peak_groups, num_groups = label(np.diff(p) < threshold)

    # iterate through groups and take the mean as peak index
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output


def detect_peaks(
    abp_signal: np.array, threshold: float = 0.3, peaks_filter=None
) -> tuple[np.array, np.array]:
    """
    Peak detection algorithm using cross correlation and threshold
    Default filter is a hyperbolic cosine function with 12 samples
    :param abp_signal: abp signal
    :param threshold: threshold for peak detection from similarity
    :param peaks_filter: qrs filter
    :return: peaks, similarity
    """
    if peaks_filter is None:
        # create default filter
        t = np.linspace(-1, 1, 12)
        peaks_filter = np.cosh(t)

    # normalize data
    abp_signal = (abp_signal - abp_signal.mean()) / abp_signal.std()

    # calculate cross correlation
    similarity = np.correlate(abp_signal, peaks_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return abp_signal[similarity > threshold].index, similarity
