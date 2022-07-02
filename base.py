"""
Deep Hedging Base.
------------------
Import this file in all deep hedging files.

from .base import Logger, Config, tf, tfp, dh_type

June 30, 2022
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config # NOQA
from cdxbasics.prettydict import PrettyOrderedDict as pdct # NOQA
from cdxbasics.util import isAtomic
import numpy as np
import tensorflow as tf
import math as math
import tensorflow_probability as tfp # NOQA
_log = Logger(__file__)

# -------------------------------------------------
# Manage tensor flow
# -------------------------------------------------

version = [ int(x) for x in tf.__version__.split(".") ]
version = version[0]*100+version[1]
_log.verify( version >= 203, "Tensor Flow version 2.3 required. Found %s", tf.__version__)

print("Tensorflow version %s" % tf.__version__)

dh_dtype = tf.float32
tf.keras.backend.set_floatx(dh_dtype.name)

# -------------------------------------------------
# TF <--> NP
# -------------------------------------------------

def tfCast( x, native = True, dtype=None ):
    """
    Casts an object or a collection of objecyts iteratively into tensors.
    Turns all custom dictionaries into dictionaries.
    
    Parameters
    ----------
        x
            object. Can be list of lists of dicts of numpys etc
                - numpy arrays become tenors
                - tensors will be cast to dtype, if required
                - atomic variables become tensor constants
        native : bool, optional
            True
                - lists of x's becomes lists of tensors
                - dicts of x's becomes dicts of tensors
            False:
                - lists of x's becomes lists of npCast(x)'s
                - dicts of x's becomes dicts of npCast(x)'s
                            
        dtype : tf.DType, optional
            Overwrite dtype
            
    Returns
    -------
        tensors.
    """
    if isinstance(x, tf.Tensor):
        return x if ( dtype is None or x.dtype == dtype ) else tf.convert_to_tensor( x, dtype=dtype )
    if isinstance(x, np.ndarray):
        return tf.convert_to_tensor( x, dtype=dtype )
    if isAtomic(x):
        return tf.constant( x, dtype=dtype )     
    if isinstance(x, dict):
        d = { _ : tfCast(x[_], dtype=dtype) for _ in x }
        return d if native or (type(x) == 'dict') else x.__class__(d)
    if isinstance(x, list):
        l = [ tfCast(x[_], dtype=dtype) for _ in x ]
        return l if native or (type(l) == 'list') else x.__class__(l)
    
    _log.verify( False, "Cannot convert object of type '%s' to tensor", x.__class__.__name__)
    return None

def tf_dict(**kwargs):
    """ Return a (standard) dictionary of tensors """
    return tfCast(kwargs)

def npCast( x, dtype=None ):
    """
    Casts an object or a collection of objecyts iteratively into numpy arrays.
    
    Parameters
    ----------
        x
            object. Can be list of lists of dicts of tensors etc
                - tensors become numpy arrays (copies !)
                - numpy arrays will be cast into dtype if necessaryt
                - atomic variables become arrays with shape ()
                - lists of x's becomes lists of npCast(x)'s
                - dicts of x's becomes dicts of npCast(x)'s
            
        dtype : tf.DType, optional
            Overwrite dtype
            
    Returns
    -------
        numpys.
    """
    if isinstance(x, tf.Tensor):
        return np.asarray( x, dtype=dtype )
    if isinstance(x, np.ndarray):
        return np.asarray( x, dtype=dtype )
    if isAtomic(x):
        return np.array(x, dtype=dtype )
    if isinstance(x, dict):
        d  = { _ : npCast(x[_], dtype=dtype) for _ in x }
        return d if type(x) == 'dict' else x.__class__(d)
    if isinstance(x, list):
        l = [ npCast(x[_], dtype=dtype) for _ in x ]
        return l if type(l) == 'list' else x.__class__(l)
    
    return  np.asarray( x, dtype=dtype )

# -------------------------------------------------
# TF flattening
# -------------------------------------------------

@tf.function
def tf_back_flatten( tensor : tf.Tensor, dim : int ) -> tf.Tensor:
    """
    Flattens a tenosr while keeping the first 'dim'-1 axis the same.
    
    x = tf.Tensor( np.array((16,8,4,2)) )
    tf_back_flatten( x, dim = 1)   --> shape [16*8*4*2]
    tf_back_flatten( x, dim = 2)   --> shape [16,8*4*2]
    ...
        
    Parameters
    ----------
        tensor : tf.Tensor
            A tensor
        dim : int
            max dimension of the flattened tensor.
            
    Returns
    -------
        Flat tensor.
    """
    _log.verify( dim > 0 and dim <= len(tensor.shape), "'dim' most be positive and not exceed dimension of tensor, %ld", len(tensor.shape))     
    if len(tensor.shape) > dim:
        splits = [ tf_back_flatten( tensor[...,_], dim=dim ) for _ in range(tensor.shape[-1]) ]
        tensor = tf.concat( splits, axis=-1 )    
    return tensor

@tf.function
def tf_make_dim( tensor : tf.Tensor, dim : int ) -> tf.Tensor:
    """
    Ensure a tensor as a given dimension by either flattening at the end to
    reduce diemnsions, or adding tf.newaxis to increase them
    
    Parameters
    ----------
        tensor : tf.Tensor
            A tensor
        dim : int
            target dimension of the flattened tensor.
            
    Returns
    -------
        Flat tensor.
    """
    if len(tensor.shape) > dim:
        return tf_back_flatten(tensor,dim)
    while len(tensor.shape) < dim:
        tensor = tensor[...,tf.newaxis]
    return tensor

# -------------------------------------------------
# Basic arithmetics
# -------------------------------------------------

def mean( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted mean """
    assert P.shape == w.shape, "Bad shapes: %s != %s" % (P.shape, w.shape)
    P = np.asarray(P)
    m = np.sum( P * w, axis=axis )
    assert not np.isnan(m), "Internal error: %g\n P = %s\n w = %s" % (m, P, w)
    return m
        
def var( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted variance """
    m = mean(P,w,axis)
    v = np.sum( P * (( w - m ) ** 2) , axis=axis )
    assert v >= 0., "Internal error: %g vs %g" % (np.sum( P * ( w** 2 ), axis=axis ), m**2)
    return v
        
def std( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted std deviation """
    s = math.sqrt( var(P,w,axis)  )
    assert not np.isnan(s), "Internal error: %g" % s
    return s
       
def err( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted std error """
    assert P.shape[0] > 0, "No P?"
    e = std(P,w,axis=axis) / math.sqrt( float(P.shape[0]) )
    assert not np.isnan(e), "Internal error: %g" % e
    return e

def mean_bins( x : np.ndarray, bins : int, weights = None ) -> np.ndarray:
    """
    Return a vector of means of bin s of x
    Typical use case is computing the mean over percentiles, e.g.
    
        x = np.sort(x)
        b = mean_bins(x, 9)
        
    The resulting 'b' essentially represents E[X|ai<X<ai+1] with ai = ith/10 percentile
    
    Parameters
    ----------
        x : vector
        bins : int
            Number of bins
        axis : int
            Along which axis
    Returns
    -------
        Numpy array of bins.
    """
    def w_mean(x,p,i1,i2):
        return np.mean( x[i1:i2] ) if p is None else np.sum( (x*p)[i1:i2] ) / np.sum( p[i1:i2] )

    x    = np.asarray(x)
    l    = len(x)
    assert len(x.shape) == 1, "Need to extend to more axes"
    if l <= bins:
        return x
    assert bins > 0, "'bins' must be positive"
    if bins == 1:
        return w_mean(x,weights,0,l)

    ixs  = np.linspace(0,l,bins+1, endpoint=True, dtype=np.int32)
    bins = np.array( [ w_mean(x,weights,ixs[i],ixs[i+1]) for i in range(bins) ] )
    return bins
    
def mean_cum_bins( x : np.ndarray, bins : int, weights = None ) -> np.ndarray:
    """
    Return a vector of means of bins of x
    
    Parameters
    ----------
        x : vector
        bins : int
            Number of bins
        axis : int
                Along which axis
    Returns
    -------
        Nunpy array of bins.
    """
    def w_mean(x,p,i):
        return np.mean( x[0:i] ) if p is None else np.sum( (x*p)[0:i] ) / np.sum( p[0:i] )

    x    = np.asarray(x)
    l    = len(x)
    assert len(x.shape) == 1, "Need to extend to more axes"
    if l <= bins:
        return x
    assert bins > 0, "'bins' must be positive"
    if bins == 1:
        return w_mean(x,weights,l)

    ixs  = np.linspace(0,l,bins+1, endpoint=True, dtype=np.int32)
    bins = np.array( [ w_mean(x,weights,ixs[i+1]) for i in range(bins) ] )
    return bins

def perct_exp( x : np.ndarray, lo : float, hi : float, weights : np.ndarray = None ) -> np.ndarray:
    """
    Compute the expectation over a percentile i.e. it will
    sort x and then compute np.mean( x[:len*lo] ) and np.mean( x[hi*len:] ).
    If a matrix instead of vector is given it will assume that
    the first dim is the sample dimension
    """    
    lo   = float(lo)
    hi   = float(hi)
    assert lo >= 0. and lo <= 1., "Percentiles must be betwee 0 and 1, not %g" % lo
    assert hi >= 0. and hi <= 1., "Percentiles must be betwee 0 and 1, not %g" % hi
    
    if len(x.shape) == 2:
        return np.array( [ perct_exp( x[:,i], lo=lo, hi=hi, weights=weights ) for i in range(x.shape[1]) ] )
    assert len(x.shape) == 1, "Can only handle matrices or vectors"
    
    ixLo    = min( math.ceil(  x.shape[0] * lo ), x.shape[0]-1 )
    ixHi    = max( math.floor( x.shape[0] * hi ), 0 )
    ixs     = np.argsort(x)
    x       = x[ixs]
    weights = weights[ixs] if not weights is None else None
        
    return  np.array( [ np.sum( (weights*x)[:ixLo] ) / np.sum( weights[:ixLo] ) if not weights is None else np.mean( x[:ixLo] ), \
                        np.sum( (weights*x)[ixHi:] ) / np.sum( weights[ixHi:] ) if not weights is None else np.mean( x[ixHi:] ) ] )
               
# -------------------------------------------------
# Generic basics
# -------------------------------------------------

def fmt_seconds( seconds : int ) -> str:
    """ Print nice format string for seconds """
    if seconds < 60:
        return "%lds" % seconds
    if seconds < 60*60:
        return "%ld:%02ld" % (seconds//60, seconds%60)
    return "%ld:%02ld:%02ld" % (seconds//60//60, (seconds//60)%60, seconds%60)    








    