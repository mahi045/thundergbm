import pandas as pd
import os

from zipfile import ZipFile


def unzip_ontime_data():
    for zipname in os.listdir("data/raw"):
        zippath = "data/raw/" + zipname
        with ZipFile(zippath, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall('data/unzip')


def read_holiday():
    holiday_excel_fnames = [fname for fname in os.listdir("data/raw") if fname.startswith("federal-holidays")]

    holiday_df = pd.DataFrame()
    for fname in holiday_excel_fnames:
        temp_df = pd.read_excel("data/raw/" + fname).iloc[1:, :1]
        temp_df = temp_df.rename(columns={temp_df.columns[0]: 'holiday'})
        holiday_df = holiday_df.append(temp_df)

    holiday_df = holiday_df[holiday_df['holiday'] != '© Calendarpedia®   www.calendarpedia.com']
    holiday_df.to_csv('data/holiday_2015-2020.csv', index=False)

    return holiday_df


def read_airport():
    airport_columns = [
        'name',
        'city',
        'country',
        'iata',
        'icao',
        'lat',
        'lon',
        'alt',
        'tz',
        'dst',
        'tzdb'
    ]
    airport_df = pd.read_json('data/raw/airport_list.json', orient='values')
    airport_df.columns = airport_columns
    airport_df = airport_df[['iata', 'lat', 'lon', 'alt']]
    airport_df = airport_df.reset_index()

    ignored_airport = ['PSG', 'SIT', 'GST', 'KTN', 'YAK', 'SJU', 'PPG', 'LIH', 'JNU', 'PSE', 'STT', 'KOA', 'STX', 'WRG',
                       'GUM', 'BQN', 'HNL', 'CDV', 'OGG', 'ITO', 'SPN']
    airport_df = airport_df[~airport_df.iata.isin(ignored_airport)]

    airport_df.to_csv('data/airport.csv', index=False)

    return airport_df


def read_airport_with_idx():
    airport_df = pd.read_csv('data/airport.csv')

    airport_idx_df = pd.read_csv('data/raw/iata_code.txt')
    airport_idx_df.columns = ['iata']
    airport_idx_df['iata_idx'] = airport_idx_df.index

    airport_df = airport_df.merge(airport_idx_df, on='iata')

    return airport_df


def to_tgbm_input(df, writePath):
    data = df[['is_delayed', 'Origin_iata_idx', 'Dest_iata_idx', 'Year', 'Month', 'DayofMonth', 'Quarter',
               'IATA_CODE_Reporting_Airline', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance',
               'ArrivalDelayGroups', 'is_holiday']]
    data = data.astype(str)
    for idx, col in enumerate(data.columns):
        if idx == 0:
            continue
        data[col] = data[col].apply(lambda x: f'{idx}:{x}')
    data.to_csv(writePath, sep=' ', index=False, header=False)


def merge_data():
    holiday_df = pd.read_csv('data/holiday_2015-2020.csv')
    holiday_df['holiday'] = pd.to_datetime(holiday_df['holiday'], format='%B %d, %Y').dt.strftime("%Y-%m-%d")
    holiday_df['is_holiday'] = 1

    airport_df = read_airport_with_idx()

    columns = [
        'Year',
        'Quarter',
        'Month',
        'DayofMonth',
        'DayOfWeek',
        'FlightDate',
        'IATA_CODE_Reporting_Airline',
        'Origin',
        'Dest',
        'CRSDepTime',
        'DepDelayMinutes',
        'DepDel15',
        'CRSArrTime',
        'ArrDelayMinutes',
        'ArrDel15',
        'CRSElapsedTime',
        'Distance',
        'ArrivalDelayGroups'
    ]

    for fname in os.listdir('data/unzip'):
        ontime_data = pd.read_csv('data/unzip/' + fname)[columns]
        ontime_data = ontime_data.merge(holiday_df, left_on='FlightDate', right_on='holiday', how='left')
        ontime_data['is_holiday'] = ontime_data['is_holiday'].fillna(0)

        ontime_data = ontime_data.merge(airport_df.rename(columns={x: 'Origin_' + x for x in airport_df.columns}),
                                        left_on='Origin', right_on='Origin_iata')
        ontime_data = ontime_data.merge(airport_df.rename(columns={x: 'Dest_' + x for x in airport_df.columns}),
                                        left_on='Dest', right_on='Dest_iata')

        ontime_data = ontime_data.drop(['holiday', 'Origin_iata', 'Dest_iata'], axis=1)

        ontime_data['is_delayed'] = ((ontime_data['DepDel15'] + ontime_data['ArrDel15']) >= 1).astype(int)

        ontime_data.to_csv('data/processed/' + fname, index=False)
        # to_tgbm_input(ontime_data, 'data/tgbm_input/'+fname.split('.')[0] + '.txt')


# read_holiday()
# read_airport()
merge_data()
