import pandas as pd
import numpy as np
import datetime
import xlrd
from typing import List, Tuple


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
    Gives how many 200ms steps are between one date and the other
    :param timestamp1: later date
    :param timestamp2: earlier date
    :return: number of 200ms steps between dates
    """
    diff = timestamp1 - timestamp2
    diff_milliseconds = diff.total_seconds() * 1000

    return int(diff_milliseconds)//5


def fill_missing_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes Data Frame with column names 'TimeSteps' and 'abp[mmHg]', where time steps are 200ms steps from 0
    and abp[mmHg] are values of a measurement. It fills gaps between time steps filling abp[mmHg] with None
    :param df: Data Frame with column names 'TimeSteps' and 'abp[mmHg]'
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
                'abp[mmHg]': np.nan
            }
            rows.append(new_row)
            df.iat[i, df.columns.get_loc('TimeSteps')] += 1
            diff -= 1

    rows.append(df.iloc[-1].to_dict())
    new_data = pd.DataFrame(rows)
    return new_data


def read_data(data: str) -> pd.DataFrame:
    """
    Reads csv data from icm+ program and adjusts it to later use.
    It reads only columns called: 'DateTime' and 'abp[mmHg]'
    It expects that data in csv file is separated by ';' and float numbers are defined using coma instead of dot.
    It changes icm+ date format column into 'TimeSteps', where ich step means 200ms gap
    It also fills the missing steps with None values.
    :param data: measurements performed using icm+ with columns called 'DateTime' and 'abp[mmHg]
    :return: adjusted Data Frame for further operations, with column names 'TimeSteps' and 'abp[mmHg]',
             where time steps are 200ms steps from 0 and abp[mmHg] are values of a measurement.
    """
    df = pd.read_csv(data, sep=';')
    try:
        df = df[['DateTime', 'abp[mmHg]']]
    except KeyError:
        print('Wrong columns names!')

    df = df.apply(lambda x: x.str.replace(',', '.'))
    df = df.apply(lambda x: [float(num) for num in x])
    df['DateTime'] = [icmp_dateformat_to_datetime(date) for date in df['DateTime']]
    df['DateTime'] = [timestamp_diff(date, df['DateTime'][0]) for date in df['DateTime']]
    df.rename(columns={'DateTime': "TimeSteps"}, inplace=True)
    df = fill_missing_steps(df)
    df['TimeSteps'] = df['TimeSteps'].astype(int)
    return df


def find_longest_segments(df: pd.DataFrame, n: int = 1) -> List[Tuple[pd.DataFrame, Tuple[int, int]]]:
    """
    Takes Data Frame with column names 'TimeSteps' and 'abp[mmHg]'.
    It finds in that data the longest segments where only single
    None values are allowed and are filled by mean of it 2 neighbors.
    :param df: Data Frame with column names 'TimeSteps' and 'abp[mmHg]'
    :param n: number of segments that function has to return
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

    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)

    results = []

    for seq in sequences[:n]:
        steps = []
        vals = []
        for step, val in seq:
            steps.append(step)
            vals.append(val)
        data = {'abp[mmHg]': vals}
        temp_df = pd.DataFrame(data)

        # If None value is at the beginning or at the end of sequence it is being deleted
        if pd.isna(temp_df['abp[mmHg]'].iloc[0]):
            temp_df = temp_df.iloc[1:]
        if pd.isna(temp_df['abp[mmHg]'].iloc[-1]):
            temp_df = temp_df.iloc[:-1]

        temp_df['abp[mmHg]'] = temp_df['abp[mmHg]'].interpolate()

        results.append((temp_df, (steps[0], steps[-1])))

    return results
