import pandas as pd
import numpy as np
import datetime
import xlrd
from typing import List, Tuple
from scipy.signal import find_peaks


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

    return int(diff_milliseconds)//5


def fill_missing_steps(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Takes Data Frame with column names 'TimeSteps' and 'Values', where TimeSteps are 5ms steps between values
    and Values are values of a measurement. It fills gaps between time steps filling Values with None
    :param df: Data Frame with column names 'TimeSteps' and 'Values'
    :return: Data Frame with filled gaps
    """
    rows = []
    for i in range(len(df) - 1):

        diff = df.iloc[i + 1]['TimeSteps'] - df.iloc[i]['TimeSteps']

        # deletes repetitions
        if diff != 0:
            rows.append(df.iloc[i].to_dict())

        # fills all gaps between two time steps
        while diff > 1:
            new_row = {
                'TimeSteps': df.iloc[i]['TimeSteps'] + 1,
                'Values': np.nan
            }
            rows.append(new_row)
            df.iat[i, df.columns.get_loc('TimeSteps')] += 1
            diff -= 1

    rows.append(df.iloc[-1].to_dict())
    new_data = pd.DataFrame(rows)
    return new_data


def read_data(data: str, signal_name: str) -> pd.DataFrame:
    """
    Reads csv data from icm+ program and adjusts it for later use.
    It reads only columns called: DateTime and 'signal_name'
    It expects that data in csv file is separated by ';' and float numbers are defined using coma instead of dot.
    It changes icm+ date format column into 'TimeSteps', where ich step means 5ms gap
    It also fills the missing steps with None values.
    :param data: measurements performed using icm+ program
    :param signal_name: column name of signal values
    :return: adjusted Data Frame for further operations, with column names 'TimeSteps' and 'Values',
             where time steps are 5ms steps between signal values.
    """

    df = pd.read_csv(data, sep=';')
    try:
        df = df[['DateTime', signal_name]]
    except KeyError:
        print('Wrong columns names!')

    df = df.apply(lambda x: x.str.replace(',', '.'))
    df = df.apply(lambda x: [float(num) for num in x])
    df['DateTime'] = [icmp_dateformat_to_datetime(date) for date in df['DateTime']]
    df['DateTime'] = [timestamp_diff(date, df['DateTime'][0]) for date in df['DateTime']]
    df.rename(columns={'DateTime': "TimeSteps"}, inplace=True)
    df.rename(columns={signal_name: "Values"}, inplace=True)
    df = fill_missing_steps(df)
    df['TimeSteps'] = df['TimeSteps'].astype(int)
    return df


def find_longest_segments(df: pd.DataFrame, n: int = 1) -> List[Tuple[np.array, Tuple[str, str]]]:
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
    for val, step in zip(df['Values'], df['TimeSteps']):
        if not pd.isna(val) or (pd.isna(val) and (not current_sequence or not pd.isna(current_sequence[-1][1]))):
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
                signal[i] = np.mean([signal[i-1], signal[i+1]])

        # formatting time to minutes:seconds
        time_start = format_time(steps[0]*0.005)
        time_end = format_time(steps[-1]*0.005)

        results.append((signal, (time_start, time_end)))

    return results

def find_short_segments(df: pd.DataFrame, n: int = 1, time: int = 60000,) -> List[Tuple[pd.DataFrame, Tuple[int, int]]]:
    """
    Takes Data Frame with column names 'TimeSteps' and 'abp[mmHg]'.
    It finds in that data segments of 'time' length without overlapping.
    Only single None values are allowed and are filled by mean of it 2 neighbors.
    
    :param df: Data Frame with column names 'TimeSteps' and 'abp[mmHg]'
    :param n: number of segments that function has to return
    :param time: length of segment in 5ms timestamps, default 60000 (5min)
    :return: Sorted by longest segments list of length n of tuples.
             First value is a DataFrame with 'abp[mmHg]' column.
             Second value is tuple with starting and ending step for that sequence
    """
    sequences = []
    current_sequence = []
    for val, step in zip(df['abp[mmHg]'], df['TimeSteps']):
        if not pd.isna(val) or (pd.isna(val) and (not current_sequence or not pd.isna(current_sequence[-1][1]))):
            current_sequence.append((step, val))
        else:
            sequences.append(current_sequence.copy())
            current_sequence = []

    if current_sequence:
        sequences.append(current_sequence)


def get_rr_intervals(signal: np.array, height: float | int) -> np.array:
    """
    Takes signal and calculates rr intervals
    :param signal: values of signal in form of np array vector
    :param height: minimum height above which will be registered peak
    :return: rr intervals
    """

    peaks, _ = find_peaks(signal, height)
    return np.diff(peaks)

def format_time(seconds: float | int) -> str:
    """
    Formats time from seconds to minutes:seconds
    :param seconds: number of seconds
    :return: time in minutes:seconds format
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds) % 60
    return f'{minutes}:{remaining_seconds:02d}'

def describe_file(df: pd.DataFrame, name: str):
    """
    Prints information about file
    :param df: Data frame with column names 'TimeSteps' and 'Values'
    :param name: name of the file
    """
    print(f'File name: {name}')
    print(f'File size: {len(df)}')
    print(f'Total measurement time: {format_time(seconds=len(df) * 0.005)} min')
    print(f'Percent of missing data: {(len(df[df["Values"].isna() == True]) / len(df) * 100):.2f}%')

def describe_segment(signal: np.array, time_interval: Tuple[str, str]):
    """
    Prints information about segment
    :param signal: values of signal in form of np array vector
    :param time_interval: tuple with starting and ending time for that sequence
    """
    print(f'Segment size: {len(signal)}')
    print(f'Segment length: {format_time(seconds=len(signal) * 0.005)} min')
    print(f'Segment interval: {time_interval[0]} min - {time_interval[1]} min')
