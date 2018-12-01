def df_iso(df_in, iso_df, alpha2=True, remove_unmatched=True):
    df = df_in.join(iso_df)
    if alpha2:
        df = df.set_index('alpha-2')
        df = df.drop(['alpha-3'], axis=1)
    else:
        df = df.set_index('alpha-3')
        df = df.drop(['alpha-2'], axis=1)
    while remove_unmatched and float('nan') in list(df.index):
        df = df.drop([float('nan')], axis=0)
    df = df.drop(['ISO Country Name'], axis=1)
    df.index.name = 'Country'
    return df