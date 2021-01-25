def read_dat(file, sep=' ', encoding="utf-8"):
    """Read data from a csv file.
    """
    
    matrix = []
    with open(file, 'r', encoding=encoding) as fp:
        for line in fp.readlines():
            matrix.append([i for i in line.split(sep)])
    return matrix