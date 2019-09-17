# ------------------------------------------------------------------------------
# Name:        dictarray
# Purpose:     extended recarray allowing to pass dicts and list of dicts
# Author:      atiefenauer
#
# Created:     01.08.2015
# Copyright:   (c) atiefenauer 2014
# Licence:     Sensirion AG
# ------------------------------------------------------------------------------
# !/usr/bin/env python
__author__ = "atiefenauer"
__copyright__ = "(c) Sensirion AG 2015"
__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


import numpy as np
rarray = np.rec.array
recarray = np.recarray
atleast_1d = np.atleast_1d
ndarray = np.ndarray
concatenate = np.concatenate
array = np.array
in1d = np.in1d

from fileio import dictarrayreader, isfileobj
from matplotlib.mlab import rec2txt


def iterdicts(rec):
    """
    >>> [r for r in iterdicts(dictarray())]
    []
    >>> [r for r in iterdicts(dictarray([1,2]))]
    [{'f0': 1, 'f1': 2}]
    """
    return (dict(zip(*(rec.dtype.names, ii))) for ii in rec.ravel())


def iteritems(rec):
    """
    >>> [r for r in iteritems(dictarray())]
    []
    >>> [r for r in iteritems(dictarray([1,2]))]
    [('f0', array([1])), ('f1', array([2]))]
    """
    if rec.dtype.names:
        return ((n, rec[n]) for n in rec.dtype.names)
    else:
        return iter([])


class DictArray(recarray):
    """
    >>> d = DictArray(shape=(0,), dtype='V4')
    >>> d  # doctest: +NORMALIZE_WHITESPACE
    DictArray([],
          dtype='|V4')
    >>> print d
    []
    >>> rarray([[1,2],[3,4]]).view(DictArray)  # doctest: +NORMALIZE_WHITESPACE
    DictArray([(1, 2), (3, 4)],
          dtype=[('f0', '<i4'), ('f1', '<i4')])
    """
    def __new__(cls, **kwargs):
        return recarray.__new__(cls, **kwargs)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __str__(self):
        if self.names and self.size:
            table_ = rec2txt(self, padding=3, precision=4)
            #line = '.'*max(map(len, table_.partition('\n')))
            #return '\n'.join([line, table_, line])
            return table_
        else:
            return '[]'

    @property
    def names(self):
        return self.dtype.names if self.dtype.names else tuple([])

    def iterfields(self):
        return (f[1] for f in iteritems(self))

    def fielddict(self):
        return dict(iteritems(self))

    def iterdicts(self):
        return iterdicts(self)

    def itemindex(self, **kwargs):
        ids = list()
        for key, val in kwargs.items():
            col = self.field(key)
            val = array(val).ravel()
            idx = np.nonzero(in1d(col, val))
            ids.append(idx)
        return tuple(concatenate(i) for i in zip(*ids))

    def extract(self, field=None, **kwargs):
        indices = self.itemindex(**kwargs)
        return self.take(indices)[field]

    def itemexists(self, **kwargs):
        return bool(self.itemindex(**kwargs))


def readdictarray(*args, **kwargs):
    try:
        read = dictarrayreader(*args, **kwargs)
    except IndexError:
        read = dictarray()
    return read


def dictarray(obj=None, **kwargs):
    """
    >>> print dictarray()
    []
    >>> print dictarray([])
    []
    >>> d0 = dictarray(names='a,b')
    >>> print d0
    []
    >>> print d0.dtype
    [('a', 'V4'), ('b', 'V4')]
    >>> d1 = dictarray([1,2], names='a,b')
    >>> print d1  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
    >>> print d1.dtype
    [('a', '<i4'), ('b', '<i4')]
    >>> d2 = dictarray({'b':2, 'a':1}, names='a,b')
    >>> print d2  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
    >>> print d2.dtype
    [('a', '<i4'), ('b', '<i4')]
    >>> d3 = dictarray([{'a':1, 'b':2}])
    >>> print d3  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
    >>> print d3.dtype
    [('a', '<i4'), ('b', '<i4')]
    >>> dictarray([{'a':1, 'b':2}], names='a,c')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...
    """
    dtype = kwargs.get('dtype', None)

    names = kwargs.get('names', None)
    if isinstance(names, str):
        names = names.split(',')
    names = list(names) if names else names

    if isinstance(obj, (list, tuple, dict)) and len(obj) == 0:
        obj = None

    if obj is None:  # TODO: rather check empty here: "not bool(arr.size)"
        kwargs.update({'shape': (0,)})  # for correct boolean not shape=() !
        if not dtype and 'formats' not in kwargs:  # or not 'names' in kwargs:
            if names:
                kwargs.update({'formats': ','.join(['V4']*len(names))})
            else:
                kwargs.update({'dtype': 'V4'})

    # iterables of dicts to dicts of lists
    elif isinstance(obj, (list, tuple)) \
            and isinstance(obj[0], dict):
        obj = dict([(n, [d[n] for d in obj])
                    for n in obj[0].keys()])

    # dicts of lists to (names and lists of tuples)
    if isinstance(obj, dict):
        key, val = zip(*obj.items())  # tuple of names, tuple of cols
        knames = map(str, key)  # make names strings in case they are not
        if names:
            # check whether dict.keys match the names in case names is given
            if not set(names) == set(knames):
                raise ValueError('names must be ' + ','.join(map(str, knames)))
            # make sure data is sorted in the order as in names
            val = [obj[n] for n in names]
        else:
            names = knames
            kwargs.update(names=names)
        # create records out of tuple of cols
        if all(map(lambda v: isinstance(v, (list, tuple)), val)):
            obj = zip(*val)
        else:  # if dictionary values are scalars
            obj = val

    elif isfileobj(obj):
        obj = readdictarray(obj)

    elif isinstance(obj, ndarray):
        if dtype:
            descr = dtype if isinstance(dtype, (list, tuple)) else dtype.descr
            names = zip(*descr)[0]
        if names:
            obj = obj[names]

    a = rarray(obj, **kwargs)
    return atleast_1d(a).view(DictArray)


if __name__ == '__main__':
    import doctest
    doctest.testmod()