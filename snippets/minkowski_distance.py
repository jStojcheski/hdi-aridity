def minkowski_distance(s1, s2, degree):
    return (np.abs(s1 - s2) ** degree).sum() ** (1/degree)