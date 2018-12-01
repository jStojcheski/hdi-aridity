def vectorize_cru_data(data, periods):
    time = list()
    values = list()
    for year in data.index:
        for period in periods:
            current_period = str(year)
            current_period += '-{}'.format(period) if period != 'ANN' else ''
            time.append(current_period)
            values.append(data[period][year])
    series = pd.Series(data=values, index=time)
    return series