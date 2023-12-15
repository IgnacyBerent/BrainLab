import pandas as pd
import numpy as np
import datetime
import xlrd
from typing import List, Tuple
import scipy.signal as sp_sig
import scipy.fft as sp_fft


def icmp_dateformat_to_datetime(icmp_time_mark: float) -> datetime:
    """
    Converts icm+ dateformat to datetime
    :param icmp_time_mark: date in icm+ format
    :return: datetime
    """
    datetime_date = xlrd.xldate_as_datetime(icmp_time_mark, 0)
    datetime_date = datetime_date + datetime.timedelta(hours=1)
    return datetime_date


def timestamp_diff(timestamp1: datetime, timestamp2: datetime) -> int:
    """
    Gives how many 5ms steps are between one date and the other
    :param timestamp1: later date
    :param timestamp2: earlier date
    :return: number of 5ms steps between dates
    """
    diff = timestamp1 - timestamp2
    diff_milliseconds = diff.total_seconds() * 1000

    return int(diff_milliseconds) // 5


def fill_missing_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes Data Frame with column names 'TimeSteps' and 'Values', where TimeSteps are 5ms steps between values
    and Values are values of a measurement. It fills gaps between time steps filling Values with None
    :param df: Data Frame with column names 'TimeSteps' and 'Values'
    :return: Data Frame with filled gaps
    """
    rows = []
    for i in range(len(df) - 1):
        diff = df.iloc[i + 1]["TimeSteps"] - df.iloc[i]["TimeSteps"]

        # deletes repetitions
        if diff != 0:
            rows.append(df.iloc[i].to_dict())

        # fills all gaps between two time steps
        while diff > 1:
            new_row = {"TimeSteps": df.iloc[i]["TimeSteps"] + 1, "Values": np.nan}
            rows.append(new_row)
            df.iat[i, df.columns.get_loc("TimeSteps")] += 1
            diff -= 1

    rows.append(df.iloc[-1].to_dict())
    new_data = pd.DataFrame(rows)
    return new_data


def read_data(data: str, signal_name: str, sep: str = ",") -> pd.DataFrame:
    """
    Reads csv data from icm+ program and adjusts it for later use.
    It reads only columns called: DateTime and 'signal_name'
    It expects that data in csv file is separated by ',' and float numbers are defined using coma instead of dot.
    It changes icm+ date format column into 'TimeSteps', where ich step means 5ms gap
    It also fills the missing steps with None values.
    :param data: measurements performed using icm+ program
    :param signal_name: column name of signal values
    :param sep: separator in csv file, default ','
    :return: adjusted Data Frame for further operations, with column names 'TimeSteps' and 'Values',
             where time steps are 5ms steps between signal values.
    """

    df = pd.read_csv(data, sep=sep)
    try:
        df = df[["DateTime", signal_name]]
    except KeyError:
        raise KeyError(f"Wrong signal name in {signal_name}!")

    if sep != ",":
        try:
            df = df.apply(lambda x: x.str.replace(",", "."))
        except AttributeError:
            raise AttributeError("Wrong separator!")

    df = df.apply(lambda x: [float(num) for num in x])
    df["DateTime"] = [icmp_dateformat_to_datetime(date) for date in df["DateTime"]]
    df["DateTime"] = [
        timestamp_diff(date, df["DateTime"][0]) for date in df["DateTime"]
    ]
    df.rename(columns={"DateTime": "TimeSteps"}, inplace=True)
    df.rename(columns={signal_name: "Values"}, inplace=True)
    df = fill_missing_steps(df)
    df["TimeSteps"] = df["TimeSteps"].astype(int)
    return df


def find_longest_segments(
    df: pd.DataFrame, n: int = 1
) -> List[Tuple[np.array, Tuple[str, str]]]:
    """
    Takes Data Frame with column names 'TimeSteps' and 'Values'.
    It finds in that data the longest segments where only single
    None values are allowed and are filled by mean of it 2 neighbors.
    :param df: Data Frame with column names 'TimeSteps' and 'Values'
    :param n: number of segments that function has to return
    :return: Sorted by longest segments list of length n of tuples.
             First value is a signal.
             Second value is tuple with starting and ending time for that sequence
    """
    sequences = []
    current_sequence = []
    for val, step in zip(df["Values"], df["TimeSteps"]):
        if not pd.isna(val) or (
            pd.isna(val)
            and (not current_sequence or not pd.isna(current_sequence[-1][1]))
        ):
            current_sequence.append((step, val))
        else:
            sequences.append(current_sequence.copy())
            current_sequence = []

    if current_sequence:
        sequences.append(current_sequence)

    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)

    results = []

    for seq in sequences[:n]:
        steps = []
        vals = []
        for step, val in seq:
            steps.append(step)
            vals.append(val)
        signal = np.array(vals)

        # If None value is at the beginning or at the end of sequence it is being deleted
        if np.isnan(signal[0]):
            signal = signal[1:]
        if np.isnan(signal[-1]):
            signal = signal[:-1]

        # interpolation of None values
        for i in range(len(signal)):
            if np.isnan(signal[i]):
                signal[i] = np.mean([signal[i - 1], signal[i + 1]])

        # formatting time to minutes:seconds
        time_start = format_time(steps[0] * 0.005)
        time_end = format_time(steps[-1] * 0.005)

        results.append((signal, (time_start, time_end)))

    return results


def format_time(seconds: float | int) -> str:
    """
    Formats time from seconds to minutes:seconds
    :param seconds: number of seconds
    :return: time in minutes:seconds format
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds) % 60
    return f"{minutes}:{remaining_seconds:02d}"


def describe_file(df: pd.DataFrame, name: str):
    """
    Prints information about file
    :param df: Data frame with column names 'TimeSteps' and 'Values'
    :param name: name of the file
    """
    print(f"File name: {name}")
    print(f"File size: {len(df)}")
    print(f"Total measurement time: {format_time(seconds=len(df) * 0.005)} min")
    print(
        f'Percent of missing data: {(len(df[df["Values"].isna() == True]) / len(df) * 100):.2f}%'
    )


def describe_segment(signal: np.array, time_interval: Tuple[str, str]):
    """
    Prints information about segment
    :param signal: values of signal in form of np array vector
    :param time_interval: tuple with starting and ending time for that sequence
    """
    print(f"Segment size: {len(signal)}")
    print(f"Segment length: {format_time(seconds=len(signal) * 0.005)} min")
    print(f"Segment interval: {time_interval[0]} min - {time_interval[1]} min \n")


def files_time_analysis(files: list[pd.DataFrame]):
    """
    Prints information about mean and std of files time duration.
    :param files: list of Data Frames with column names 'TimeSteps' and 'Values'
    """

    files_time = np.array([len(file) for file in files]) * 0.005
    mean_time = np.mean(files_time)
    std_time = np.std(files_time)
    print(
        f"File time duration: ({format_time(mean_time)} ± {format_time(std_time)}) min"
    )
    print(f"Shortest file time duration: {format_time(np.min(files_time))} min")
    print(f"Longest file time duration: {format_time(np.max(files_time))} min")


def calculate_fundamental_component(signal: np.array, fs: float, low_f=0.66, high_f=3):
    """
    Calculates fundamental component of signal in given frequency range
    :param signal: signal
    :param fs: sampling frequency
    :param low_f: lower frequency range
    :param high_f: higher frequency range
    :return: fundamental frequency and its amplitude
    """
    n_fft = len(signal)
    win_fft_amp = sp_fft.rfft(
        sp_sig.detrend(signal) * sp_sig.windows.hann(n_fft), n=n_fft
    )
    corr_factor = n_fft / np.sum(sp_sig.windows.hann(n_fft))
    win_fft_amp = abs(win_fft_amp) * 2 / n_fft * corr_factor

    win_fft_f = sp_fft.rfftfreq(n_fft, d=1 / fs)
    f_low = int(low_f * n_fft / fs)
    f_upp = int(high_f * n_fft / fs)
    win_fft_amp_range = win_fft_amp[f_low:f_upp]
    fund_idx = np.argmax(win_fft_amp_range) + f_low

    fund_f = win_fft_f[fund_idx]
    fund_amp = win_fft_amp[fund_idx]

    # plt.figure()
    # plt.plot(win_fft_f, win_fft_amp)
    # plt.plot(win_fft_f[fund_idx], win_fft_amp[fund_idx], '.r')
    # plt.xlim(0, 10)
    # plt.show()

    return fund_f, fund_amp


def calculate_mean_HR(signal: np.array, fs: float = 200):
    """
    Calculates mean HR from signal
    :param signal: signal
    :param fs: sampling frequency
    """
    c_f1, amp_abp = calculate_fundamental_component(signal, fs)
    HR = c_f1 * 60
    return HR
