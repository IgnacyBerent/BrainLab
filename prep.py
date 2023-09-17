import pandas as pd
import numpy as np
import datetime
import xlrd

def icmp_dateformat_to_datetime(icmp_time_mark):
    datetime_date = xlrd.xldate_as_datetime(icmp_time_mark, 0)
    datetime_date = datetime_date + datetime.timedelta(hours=1)
    return datetime_date


def timestamp_diff(timestamp1, timestamp2):
    diff = timestamp1 - timestamp2
    diff_milliseconds = diff.total_seconds() * 1000

    return int(diff_milliseconds)//5


def fill_missing_steps(df):
    """
    Uzupełnia brakujące wartości pomiaru czasu jako NaN
    """
    rows = []

    # Iteracja po oryginalnych wierszach
    for i in range(len(df) - 1):

        # Sprawdzenie różnicy między obecnym wierszem a następnym
        diff = df.iloc[i + 1]['TimeSteps'] - df.iloc[i]['TimeSteps']

        # usuwa powtórzenia
        if diff != 0:
            rows.append(df.iloc[i].to_dict())

        while diff > 1:
            new_row = {
                'TimeSteps': df.iloc[i]['TimeSteps'] + 1,
                'abp[mmHg]': np.nan
            }
            rows.append(new_row)
            df.iat[i, df.columns.get_loc('TimeSteps')] += 1  # Aktualizacja wartości w oryginalnej ramce danych
            diff -= 1

    rows.append(df.iloc[-1].to_dict())
    new_data = pd.DataFrame(rows)
    return new_data


def read_data(data):
    df = pd.read_csv(data, sep=';')
    try:
        df = df[['DateTime', 'abp[mmHg]']]
    except KeyError:
        print('Złe nazwy kolumn!')

    df = df.apply(lambda x: x.str.replace(',', '.'))
    df = df.apply(lambda x: [float(num) for num in x])
    df['DateTime'] = [icmp_dateformat_to_datetime(date) for date in df['DateTime']]
    df['DateTime'] = [timestamp_diff(date, df['DateTime'][0]) for date in df['DateTime']]
    df.rename(columns = {'DateTime': "TimeSteps"}, inplace = True)
    df = fill_missing_steps(df)
    df['TimeSteps'] = df['TimeSteps'].astype(int)
    return df


def find_longest_segment(df: pd.DataFrame) -> pd.DataFrame:
    max_len = 0
    current_len = 0
    nan_count = 0

    current_sequence = []
    longest_sequence = []

    for val in df['abp[mmHg]']:
        if not pd.isna(val):
            current_len += 1
            nan_count = 0
            current_sequence.append(val)
        elif pd.isna(val) and nan_count == 0:
            current_len += 1
            nan_count += 1
            current_sequence.append(val)
        else:
            if current_len > max_len:
                max_len = current_len
                longest_sequence = current_sequence.copy()
            current_len = 0
            nan_count = 0
            current_sequence = []

    if current_len > max_len:
        longest_sequence = current_sequence.copy()

    data = {'abp[mmHg]': longest_sequence}
    result = pd.DataFrame(data)

    # Jeśli NaN występuje na początku lub końcu najdłuższego odcinka, usuń odpowiednie wiersze
    if pd.isna(result['abp[mmHg]'].iloc[0]):
        result = result.iloc[1:]
    if pd.isna(result['abp[mmHg]'].iloc[-1]):
        result = result.iloc[:-1]

    # Uzupełnij wartości NaN poprzez interpolację
    result['abp[mmHg]'] = result['abp[mmHg]'].interpolate()

    return result
