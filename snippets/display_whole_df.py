def display_whole_df(df):
    with pd.option_context('display.max_rows', df.shape[0], 'display.max_columns', df.shape[1]):
        display(df)
    return