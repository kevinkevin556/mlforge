def read_csv(file, sep=' ', encoding="utf-8"):
    """
    Read data from a csv file.
    """
    matrix = []
    with open(file, 'r', encoding=encoding) as fp:
        for line in fp.readlines():
            matrix.append([float(i) for i in line.split(sep)])
    return np.array(matrix)