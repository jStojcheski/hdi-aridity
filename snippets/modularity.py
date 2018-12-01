def modularity(df, groups):
    labels = set(groups)

    edge_set = set()
    value = 0

    for i in labels:
        for j in labels:
            cs1 = groups[groups == i].index
            cs2 = groups[groups == j].index

            for c1 in cs1:
                for c2 in cs2:
                    edge = f'{c1}_{c2}' if c1 < c2 else f'{c2}_{c1}'

                    if edge not in edge_set:
                        if i == j:
                            value += df.loc[c1, c2]
                        else:
                            value -= df.loc[c1, c2]

                        edge_set.add(edge)

    return value