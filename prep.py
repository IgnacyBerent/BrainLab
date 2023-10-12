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
    df.rename(columns={'DateTime': "TimeSteps"}, inplace=True)
    df = fill_missing_steps(df)
    df['TimeSteps'] = df['TimeSteps'].astype(int)
    return df


def find_longest_segments(df: pd.DataFrame, n: int = 1) -> list:
    sequences = []
    current_sequence = []

    for val in df['abp[mmHg]']:
        if not pd.isna(val) or (pd.isna(val) and (not current_sequence or not pd.isna(current_sequence[-1]))):
            current_sequence.append(val)
        else:
            sequences.append(current_sequence.copy())
            current_sequence = []

    if current_sequence:
        sequences.append(current_sequence)

    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)

    results = []

    for seq in sequences[:n]:
        data = {'abp[mmHg]': seq}
        temp_df = pd.DataFrame(data)

        # Jeśli NaN występuje na początku lub końcu najdłuższego odcinka, usuń odpowiednie wiersze
        if pd.isna(temp_df['abp[mmHg]'].iloc[0]):
            temp_df = temp_df.iloc[1:]
        if pd.isna(temp_df['abp[mmHg]'].iloc[-1]):
            temp_df = temp_df.iloc[:-1]

        # Uzupełnij wartości NaN poprzez interpolację
        temp_df['abp[mmHg]'] = temp_df['abp[mmHg]'].interpolate()

        results.append(temp_df)

    return results
