# ------------------------------------------------------------------------------
# Name:        dictarray_relational
# Purpose:     a relational version of dictarray
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
array = np.array
rarray = np.rec.array
result_type = np.result_type
concatenate = np.concatenate
atleast_1d = np.atleast_1d
format_parser = np.format_parser
unique = np.unique
vectorize = np.vectorize
in1d = np.in1d
npany = np.any

from dictarray import dictarray
from dictarray_mutable import MutableDictArray, append, FileDictArray


def isrelational(arr, primarykey=None):
    """ checks whether a table is relational, whereas a given name for a
        primarykey checks takes a specific field otherwise it entire records
        count as primarykey
    >>> isrelational(dictarray())
    True
    >>> isrelational(dictarray(), primarykey='f0')
    True
    >>> isrelational(dictarray(names='a,b'), primarykey='a')
    True
    >>> isrelational(rarray([[1,2],[1,4]]))
    True
    >>> isrelational(rarray([[1,2],[1,2]]))
    False
    >>> isrelational(rarray([[1,2],[1,4]]), primarykey='f1')
    True
    >>> isrelational(rarray([[1,2],[1,4]]), primarykey='f0')
    False
    >>> isrelational(rarray([[1,2],[1,2]]), primarykey='f0')
    False
    """
    if primarykey:
        primarykey = str(primarykey).strip()
        if arr.dtype.names:
            arr = arr.field(primarykey)
        else:
            return True
    return len(unique(arr.ravel())) == len(arr.ravel())


def can_append(a, b, primarykey=None, asstring=False, **kwargs):
    """
    >>> can_append([1,2],[1,2], 'a', names='a,b')
    False
    >>> can_append([1,2],[2,2], 'a', names='a,b')
    True
    >>> can_append([1,2],['1',2], 'a', names='a,b')
    True
    >>> can_append([1,2],['1',2], 'a', names='a,b', asstring=True)
    False
    """
    if primarykey:
        arr = dictarray(a, **kwargs)
        if arr.names:
            aval = dictarray(a, **kwargs).field(primarykey)
        else:
            aval = array([])
        brr = dictarray(b, **kwargs)
        if brr.names:
            bval = dictarray(b, **kwargs).field(primarykey)
        else:
            bval = array([])
        if asstring:
            strip = vectorize(lambda x: x.strip())
            aval = strip([str(v) for v in aval.ravel()])
            bval = strip([str(v) for v in bval.ravel()])
        return not npany(in1d(bval, aval)) and isrelational(bval)
    else:
        return True


def append_relational(a, b, primarykey=None, **kwargs):
    if can_append(a, b, primarykey=primarykey, **kwargs):
        return append(a, b, **kwargs)
    else:
        raise ValueError('Fail attempt adding records in relational array. ' +
                         'Value {v} already exists.'.format(v=dictarray(b)) +
                         'Current values for primarykey {p}'.format(p=primarykey) +
                         'are {v}'.format(v=','.join(b[primarykey])))


class RelationalDictArray(MutableDictArray):
    """
    >>> r0 = RelationalDictArray(formats='f4,i4', names='b,a')
    >>> r0.append(dict(b=2.5, a=1.1))
    >>> r0.append(dict(a=3.1, b=4.5))
    >>> print r0  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: None':
            b        a
       2.5000   1.1000
       4.5000   3.1000

    >>> r0 = RelationalDictArray(formats='f4,i4', primarykey='a', names='b,a')
    >>> r0.append(dict(b=2.5, a=1.1))
    >>> r0.append(dict(a=3.1, b=4.5))
    >>> print r0  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: a':
            b        a
       2.5000   1.1000
       4.5000   3.1000

    >>> r0 = RelationalDictArray(formats='f4,i4', primarykey='a', names='b,a')
    >>> r0.append(dict(b=2.5, a=1.1))
    >>> r0.append(dict(a=1.1, b=4.5))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...

    >>> print r0  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: a':
            b        a
       2.5000   1.1000
    """
    def __init__(self, obj=None, primarykey=None, **kwargs):
        super(RelationalDictArray, self).__init__(obj=obj, **kwargs)
        self.primarykey = primarykey

    def __str__(self):
        data = super(RelationalDictArray, self).__str__()
        return '\'primarykey: {pkey}\':\n{data}'.format(pkey=self.primarykey, data=data)

    def append(self, obj, **kwargs):
        arr = dictarray(obj, **kwargs)
        if self and can_append(self._data, arr, self.primarykey):
            super(RelationalDictArray, self).append(arr)
        elif not self and isrelational(arr, self.primarykey):
            super(RelationalDictArray, self).append(arr)
        else:
            pkey = self.primarykey
            values = '\n'.join([str(v) for v in self[self.primarykey].ravel()])
            raise ValueError('Fail attempt adding records in relational array. ' +
                             'There is already a similar entry.\n' +
                             'Current values for primarykey \'{p}\' '.format(p=pkey) +
                             'are:\n{v}'.format(v=values))

    def _get_primarykey(self):
        return self._primarykey

    def _set_primarykey(self, name):
        if name:
            name = str(name).strip()
            if self.names and name not in self.names:
                raise ValueError('wrong primarykey. Must be one of: ' +
                                 '{names} not {name}'.format(names=', '.join(self.names), name=name))
            if self:
                if not isrelational(self.field(name)):
                    raise ArithmeticError('cannot use field {f}'.format(f=name) +
                                          ' as primarykey. Contains none-unique '
                                          'items in {f}'.format(f=self.field(name)))
        self._primarykey = name

    primarykey = property(_get_primarykey, _set_primarykey)


class FileRelationalDictArray(RelationalDictArray, FileDictArray):
    """
    >>> import tempfile as tf
    >>> import os
    >>> filepath = os.path.join(tf.gettempdir(),'test.txt')
    >>> os.remove(filepath) if os.path.exists(filepath) else None
    >>> fileobj = open(filepath, mode='a+')
    >>> with FileRelationalDictArray(fileobj, formats='f4,i4', names='b,a') as f:
    ...     f.append(dict(b=2.5, a=1.1))
    ...     f.append(dict(a=3.1, b=4.5))
    ...     print f  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: None':
    '...temp\\test.txt':
            b        a
       2.5000   1.1000
       4.5000   3.1000
    >>> os.remove(filepath)

    >>> fileobj = open(filepath, mode='a+')
    >>> with FileRelationalDictArray(fileobj, primarykey='a', formats='f4,i4', names='b,a') as f:
    ...     f.append(dict(b=2.5, a=1.1))
    ...     f.append(dict(a=3.1, b=4.5))
    ...     print f  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: a':
    '...temp\\test.txt':
            b        a
       2.5000   1.1000
       4.5000   3.1000
    >>> f.file.close()
    >>> os.remove(filepath)


    >>> fileobj = open(filepath, mode='a+')
    >>> with FileRelationalDictArray(fileobj, primarykey='a', formats='f4,i4', names='b,a') as f:
    ...     f.append(dict(b=2.5, a=1.1))
    ...     f.append(dict(a=1.1, b=4.5))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...
    >>> print f  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    'primarykey: a':
    '...temp\\test.txt':
            b        a
       2.5000   1.1000
    >>> os.remove(filepath)
    """
    def __init__(self, *args, **kwargs):
        super(FileRelationalDictArray, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

