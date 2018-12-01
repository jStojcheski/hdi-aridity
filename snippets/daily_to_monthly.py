days_month = dict({'JAN': 31, 'FEB': 28, 'MAR': 31, 'APR': 30,
                    'MAY': 31, 'JUN': 30, 'JUL': 31, 'AUG': 31,
                    'SEP': 30, 'OCT': 31, 'NOV': 30, 'DEC': 31})

days_quartal = {'MAM': 92, 'JJA': 92, 'SON': 91, 'DJF': 90}


def daily_to_monthly(data):
    for year in data.index:
        for period in data.columns:
            value = data[period][year]
            if value != -999:
                if period in days_month.keys():
                    value = data[period][year] * (days_month[period] +
                                                 (1 if period == 'FEB' and
                                                  calendar.isleap(year) else 0))
                elif period in days_quartal.keys():
                    value = data[period][year] * (days_quartal[period] +
                                                  (1 if period == 'DJF' and
                                                  calendar.isleap(year) else 0))
                elif period == 'ANN':
                    value = data[period][year] * (366 if calendar.isleap(year)
                                                  else 365)
            value = round(value, 1)
            data[period][year] = value
    return data