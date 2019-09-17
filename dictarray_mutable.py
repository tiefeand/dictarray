# ------------------------------------------------------------------------------
# Name:        dictarray_mutable
# Purpose:     a mutable version of dictarray
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
recarray = np.recarray
concatenate = np.concatenate
atleast_1d = np.atleast_1d
result_type = np.result_type
array = np.array
in1d = np.in1d
unique = np.unique
vectorize = np.vectorize
npany = np.any

from dictarray import dictarray, DictArray
from fileio import DictArrayWriter


def isempty(arr):
    """ checks whether a 1d-sized recarray is empty

    >>> isempty(dictarray())
    True
    >>> isempty(dictarray(names='a,b'))
    True
    >>> isempty(dictarray([1,2], names='a,b'))
    False
    """
    arr = dictarray(arr)
    return not bool(arr.size)  # works if empty has shape (0,) not (,)


def append(a, b, **kwargs):
    """ concatenates two recarrays and upcasts data according to its common
        parent type
    >>> print append(dictarray(), dictarray())
    []
    >>> print append(dictarray(), dictarray(names='a,b'))
    []
    >>> print append(dictarray(names='a,b'), dictarray())
    []
    >>> print append(dictarray(), dictarray([1,2], names='a,b'))  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
    >>> print append(dictarray([1,2]), dictarray(names='a,b'))  # doctest: +NORMALIZE_WHITESPACE
       f0   f1
        1    2
    >>> d1 = append(dictarray([1,2], names='a,b'), dictarray([4,3], names='b,a'))
    >>> print d1
    [(1, 2) (3, 4)]
    >>> print d1.dtype
    [('a', '<i4'), ('b', '<i4')]
    """
    # TODO: cast according to first of both arrays rather then to a common type
    arr = dictarray(a)
    brr = dictarray(b, **kwargs)
    if isempty(brr):
        return arr
    else:
        if isempty(arr):
            if arr.dtype.names:
                brr = dictarray(b, names=arr.dtype.names)
            return brr
        else:
            dtype = map(lambda n: (n, result_type(arr[n], brr[n])), arr.dtype.names)
            arr = atleast_1d(arr.astype(dtype, casting='unsafe'))
            brr = atleast_1d(brr.astype(dtype, casting='unsafe'))
            # TODO: can we use casting='safe' ?
            return concatenate((arr, brr))


class MutableDictArray(object):  # TODO: consider derive from Sequence, or deque
    """ a mutable version of DictArray

    >>> m0 = MutableDictArray()
    >>> print m0
    []
    >>> len(m0)
    0
    >>> bool(m0)
    False
    >>> [r for r in m0]
    []
    >>> m0.append([1,2], names='a,b')
    >>> print m0  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2

    >>> m1 = MutableDictArray(names='a,b')
    >>> print m1
    []
    >>> len(m1)
    0
    >>> bool(m1)
    False
    >>> [r for r in m1]
    []
    >>> m1.append([1,2])
    >>> print m1  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2

    >>> m2 = MutableDictArray([1,2], names='a,b')
    >>> print m2  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
    >>> len(m2)
    1
    >>> bool(m2)
    True
    >>> [r for r in m2]
    [(1, 2)]
    >>> m2.append([4,3], names='b,a')
    >>> print m2  # doctest: +NORMALIZE_WHITESPACE
       a   b
       1   2
       3   4

    >>> tuple(m2.iterfields())  # doctest: +NORMALIZE_WHITESPACE
    (array([1, 3]), array([2, 4]))
    >>> m2.fielddict()  # doctest: +NORMALIZE_WHITESPACE
    {'a': array([1, 3]), 'b': array([2, 4])}
    >>> tuple(m2.iterdicts())  # doctest: +NORMALIZE_WHITESPACE
    ({'a': 1, 'b': 2}, {'a': 3, 'b': 4})

    >>> m2.append([5,6], names='a,b')
    >>> m2.itemindex(a=5)
    (array([2]),)
    >>> m2.itemindex(a=5, b=2)
    (array([2, 0]),)
    """
    def __init__(self, obj=None, **kwargs):
        super(MutableDictArray, self).__init__()
        obj = obj.todictarray() if isinstance(obj, MutableDictArray) else obj
        self._data = dictarray(obj=obj, **kwargs)

    def __getattr__(self, arg):
        try:
            return self._data.__getattribute__(arg)
        except AttributeError:
            try:
                return self._data.__getattr__(arg)
            except AttributeError:
                raise AttributeError(
                    self.__class__.__name__ + " has no attribute '" + arg + "'")

    def __getitem__(self, *arg):
        return self._data.__getitem__(*arg)

    def __len__(self):
        return len(self._data)

    def __nonzero__(self):
        return not isempty(self._data)

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self._data) + ')'

    @property
    def names(self):
        return self._data.names

    def append(self, obj, **kwargs):
        new = append(self._data, obj, **kwargs).view(DictArray)
        self._data = new

    def todictarray(self):
        return self._data


def writedictarray(fileobj, arr, writeheader=False, **kwargs):
    arr = dictarray(arr, **kwargs)
    writer = DictArrayWriter(fileobj, arr.dtype.names)
    if writeheader:
        writer.writeheader()
    writer.writerows(arr.iterdicts())


class FileDictArray(MutableDictArray):
    """ a file version of MutableDictArray that keeps entries of a file
        synchronized with the array

    >>> import tempfile as tf
    >>> import os
    >>> filepath = os.path.join(tf.gettempdir(),'test.txt')
    >>> os.remove(filepath) if os.path.exists(filepath) else None
    >>> fileobj = open(filepath, mode='a+')
    >>> with FileDictArray(fileobj, formats='f4,i4', names='b,a') as f:
    ...     f.append(dict(b=2.5, a=1.1))
    ...     f.append(dict(a=3.1, b=4.5))
    ...     print f  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    '...temp\\test.txt':
            b        a
       2.5000   1.1000
       4.5000   3.1000
    >>> fileobj = open(filepath, mode='r')
    >>> print FileDictArray(fileobj)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    '...temp\\test.txt':
            b        a
       2.5000   1.1000
       4.5000   3.1000
    """
    def __init__(self, obj=None, **kwargs):
        super(FileDictArray, self).__init__(obj=obj, **kwargs)
        self.file = obj

    def __str__(self):
        fname = self.file.name
        data = super(FileDictArray, self).__str__()
        return '\'{fname}\':\n{data}'.format(fname=fname, data=data)

    def __enter__(self):
        self.file.__enter__()
        return self

    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        self.file.close()
        return result

    def append(self, obj, **kwargs):
        names = self.names if self.names else None
        arr = dictarray(obj, names=names)
        headerflag = isempty(self)
        super(FileDictArray, self).append(arr, **kwargs)
        writedictarray(self.file, arr, writeheader=headerflag)


if __name__ == '__main__':
    import doctest
    doctest.testmod()