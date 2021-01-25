import operator
from collections import Counter 
from functools import reduce 
from itertools import accumulate
from scipy.special import comb


def C(n, r):
    return comb(n, r)



def H(n, r):
    return comb(n, r, repetition=True) 



def sequencial_comb(n, r):
    """ return sequencial product of combination:
            nCa * (n-a)Cb * (n-a-b)Cc * ...
        where r is a tuple (a, b, c, ...) and a+b+c+... = n
        
        Ex: 3C(1,2) = 3C1 * 2C2 = 3
        Used to calculate the coeffients of terms in a polynomial.
    """
    subtracting = list(accumulate([0] + list(r)[0:-1]))
    n_a = [n-i for i in subtracting]
    
    comb_seq = [C(n_, r_) for n_, r_ in zip(n_a, r)]
    output = reduce(operator.mul, comb_seq)
    return output



def get_poly_coef(deg=None, terms=None, count=None):
    """ Calculate the coefficients of the terms in a polynomial
    
        input: tuple or dict 
        output: int 
        
        Ex: X1^1 * X2^2 * X3^1 
        can be represented as (1, 2, 2, 3) or {1:1, 2:2, 3:1}
        or directly send count=(1, 2, 1) into this function.
    """
    if count is None:
        if type(terms) is tuple:
            count = dict(Counter(terms)).values()
        if type(terms) is dict:
            count = terms.values()

    if deg is None:
        deg = sum(count)

    if sum(count) == deg:
        coef = sequencial_comb(deg, count)
    else:
        raise ValueError("Total of counts should be equalled to the degree of polynomial.")

    return coef