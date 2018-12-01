def df_to_csv(df, path):
    df.to_csv('{}'.format(path), index_label='Country')