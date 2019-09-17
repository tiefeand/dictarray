# ------------------------------------------------------------------------------
# Name:        fileio
# Purpose:     specifying the dialect used to read and write files
#
#  Author:      atiefenauer
#
# Created:     01.08.2015
# Copyright:   (c) atiefenauer 2015
# Licence:     <your licence>
# ------------------------------------------------------------------------------
# !/usr/bin/env python
__author__ = "atiefenauer"
__copyright__ = "(c) Sensirion AG 2015"
__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


from csv import Dialect, register_dialect, DictWriter, QUOTE_MINIMAL
from tempfile import _TemporaryFileWrapper
from numpy import genfromtxt
import warnings

# csv.Dialect
DEFAULT_DELIMITER = '\t'
#DEFAULT_DELIMITER = ','
DEFAULT_DOUBLEQUOTE = True
DEFAULT_ESCAPECHAR = None
DEFAULT_LINEDETERMINATOR = '\n'  # rather than os.linesep
DEFAULT_QUOTECHAR = '"'
DEFAULT_QUOTING = QUOTE_MINIMAL
DEFAULT_SKIPINITIALSPACE = False

# csv.DictWriter
DEFAULT_RESTVAL = ''
DEFAULT_EXTRASACTION = 'raise'
DEFAULT_DIALECT = 'dictarray'
DEFAULT_RESTKEY = ['a']


class DictArrayDialect(Dialect):
    delimiter = DEFAULT_DELIMITER
    doublequote = DEFAULT_DOUBLEQUOTE
    escapechar = DEFAULT_ESCAPECHAR
    lineterminator = DEFAULT_LINEDETERMINATOR
    quotechar = DEFAULT_QUOTECHAR
    quoting = DEFAULT_QUOTING
    skipinitialspace = DEFAULT_SKIPINITIALSPACE


register_dialect('dictarray', DictArrayDialect)


class DictArrayWriter(DictWriter):
    """
    >>> import tempfile as tf
    >>> fileobj = tf.TemporaryFile()
    >>> w = DictArrayWriter(fileobj, ['a','b'])
    >>> w.writeheader()
    >>> w.writerow({'a':1,'b':2})
    """
    def __init__(self, *args, **kwargs):
        kwargs.update(dialect=kwargs.get('dialect', DEFAULT_DIALECT))
        DictWriter.__init__(self, *args, **kwargs)


def dictarrayreader(*args, **kwargs):
    """
    >>> import tempfile as tf
    >>> import os
    >>> fname = os.path.join(tf.gettempdir(),'file.txt')
    >>> with open(fname, mode='a+') as fileobj:
    ...     dictarrayreader(fileobj)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    IndexError: ...
    >>> with open(fname, mode='a+') as fileobj:
    ...     w = DictArrayWriter(fileobj, ['a','b'])
    ...     w.writeheader()
    ...     w.writerow({'a':1,'b':2})
    ...     w.writerow({'a':3.1,'b':4})
    >>> with open(fname, mode='a+') as fileobj:
    ...     print dictarrayreader(fileobj)  # doctest: +NORMALIZE_WHITESPACE
    [(1.0, 2) (3.1, 4)]
    >>> os.remove(fname)
    """
    dtype = None
    kwargs.update(dtype=kwargs.get('dtype', dtype),
                  delimiter=kwargs.get('delimiter', DEFAULT_DELIMITER),
                  names=kwargs.get('names', True))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return genfromtxt(*args, **kwargs)


def isfileobj(obj):
    """ checks whether obj is a fileobject """
    return isinstance(obj, file) or isinstance(obj, _TemporaryFileWrapper)


if __name__ == '__main__':
    pass
