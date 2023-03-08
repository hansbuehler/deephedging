"""
Keras Model base class with plenty of defaults
----------------------------------------------
* Cachable
* Keeps best state
* Optimized for async training

Feb 25, 2023
@author: hansbuehler
"""
from collections.abc import Mapping
from cdxbasics.config import Config, Int, Float
from cdxbasics.verbose import Context
from cdxbasics.logger import Logger
from cdxbasics.prettydict import PrettyDict as PrettyDict
from cdxbasics.util import uniqueHash
from cdxbasics.subdir import SubDir, uniqueFileName48, CacheMode
from deephedging.base import create_optimizer, npCast, tfCast, fmt_now, fmt_seconds
import tensorflow as tf
import numpy as np
import time as time

_log = Logger(__file__)

dtype = tf.float32

# ==========================================================================
# Model
# Cachable model
# ==========================================================================

class Model(tf.keras.Model):
    """ 
    Base class for a keras model with additionally
    * Automatic caching
    * Default tracking of progress, with support for asynchronous training

    This model assumes that the loss of the model is "linear", e.g. an expectation of a returned variable (usually 'loss')
    
    Implementation comments
    -----------------------
    The implementation here requires the model to be construted, compiled, and predicted once
    before it can be restored from a cache. This also means it is not very fast to restore
    a reasonably complex model.
    
    """
    
    def __init__(self, cache_uid       : str,
                       name            : str = None,
                       dtype           : tf.DType = None,
                       trainable       : bool = True,
                       cache_version   : str = None):
        """
        Initializes the cachable model
        ------------------------------
            cache_uid : Config or str
                Unique ID for this model for caching.
                You can pass a 'Config' object in which case it will call config_ID.cache_unique_id.
            name : str
                Name of the object
            dtype : tf.DType
                Type of the model
            trainable : bool
                Whether the model is trainable.
            cache_version : int
                Additional version for the cache. This allows updating caches even if no config changes (e.g when a bug in the code was found)
        """
        tf.keras.Model.__init__(self, name=name, dtype=dtype, trainable=trainable )

        if isinstance(cache_uid, Config):
            self._cache_unique_id = cache_uid.unique_id()
        elif isinstance(cache_uid, str):
            self._cache_unique_id = str( cache_uid )
        else:
            _log.throw("'config_ID' must be a 'str' or a 'Config'. Found type %s", type(cache_uid))
           
        cache_version           = str(cache_version) if not cache_version is None else cache_version
        if not cache_version is None: _log.verify( len(cache_version) > 0, "'cache_version' cannot be an empty string (or translate into one)")
        self._cache_version     = cache_version
        self._cache_ready       = False
        
    def __call__(self, *kargs, **kwargs ):
        """ Executes the model. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#call """
        self._cache_ready     = True
        return tf.keras.Model.__call__(self, *kargs, **kwargs )
        
    # -------------------
    # syntatic sugar
    # -------------------

    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] ) if not weights is None else 0.
    
    # -------------------
    # caching
    # -------------------

    @property
    def is_caching_ready(self):
        """
        Returns True if the object can be cached, or restored from cache.
        Essentially, that means that the model was at least call()ed once.
        """
        return self._cache_ready
    
    @property
    def has_optimizer(self):
        """
        Returns True if the object can be cached, or restored from cache.
        Essentially, that means that the model was at least call()ed once.
        """
        return not getattr( self, "optimizer", None ) is None
    
    @property
    def cache_uid( self ):
        """
        Return the unique ID for this model.
        This includes the optimizer, if the model was compiled
        """
        _, _, opt_uid = self._cache_get_optimizer()
        return self._cache_uid( opt_uid )
    
    @property
    def cache_def_directory_name( self ):
        """ Returns a descriptive name for this class which can be used as directory for the caches """
        name = str( self.__class__.__name__ )
        return name if self._cache_version is None else ( name + "/" + self._cache_version )
    
    def cache_create( self ) -> dict:
        """
        Create a dictionary which allows reconstructing the current model.
        The content of the dictionary are IDs to validate that we are reconstructing the same type of gym,
        weights of the model and the optimizer, and the last learning rate of the optimizer.
        Note: reconstruction of an optimizer state is not natively supported in TensorFlow. Below might not work perfectly.
        """
        opt_weights, opt_config, opt_uid = self._cache_get_optimizer()

        return dict( cache_id      = self._cache_uid( opt_uid ),
                     version       = self._cache_version,
                     model_uid     = self._cache_unique_id,
                     model_weights = self.get_weights(),
                     opt_uid       = opt_uid,
                     opt_config    = opt_config,
                     opt_weights   = opt_weights )
                
    def cache_restore( self, cache : dict, initial : bool ) -> bool:
        """
        Restore 'self' from cache.
        Note that we have to call() this object before being able to use this function        
        This function returns False if the cached weights do not match the current architecture.
        Note: reconstruction of an optimizer state is not natively supported in TensorFlow. Below might not work perfectly.
        """        
        version       = cache['version']
        model_uid     = cache['model_uid']
        model_weights = cache['model_weights']
        opt_uid       = cache['opt_uid']
        opt_config    = cache['opt_config']
        opt_weights   = cache['opt_weights']
        
        self_opt_weights, self_opt_config, self_opt_id = self._cache_get_optimizer()

        if version != self._cache_version:
            _log.warn( "Cache restoration error: provided cache object has version %ld vs current version %ld", version, self._cache_version)
            return False
        if model_uid != self._cache_unique_id:
            _log.warn( "Cache restoration error: provided cache object has gym ID %s vs current ID %s", model_uid, self._cache_unique_id)
            return False
        if opt_uid != self_opt_id:
            _log.warn( "Cache restoration error: provided cache object has optimizer ID %s vs current ID %s\n"\
                       "Stored configuration: %s\nCurrent configuration: %s", opt_uid, self_opt_uid, opt_config, self_opt_config)
            return False

        # load weights
        # Note that we will continue with the restored weights for the gym even if we fail to restore the optimizer
        # This is likely the desired behaviour.
        try:
            self.set_weights( model_weights )
        except ValueError as v:
            _log.warn( "Cache restoration error: provided cache gym weights were not compatible with the gym.\n%s", v)
            return False

        # currrently, 
        if initial:
            return True
        
        # optimizer
        optimizer   = getattr( self, "optimizer", None )
        if optimizer is None:
            return True    
        # set learning rate to last recoreded value
        if 'learning_rate' in opt_config:
            self.optimizer.learning_rate = opt_config['learning_rate']
        # restore weights
        try:
            self.optimizer.set_weights( opt_weights )
        except ValueError as v:
            ex = optimizer.get_weights()
            
            isTF211 = getattr(self.optimizer,"get_weights",None) is None
            isTF211 = "" if not isTF211 else "\nCode is running TensorFlow 2.11 or higher for which tf.keras.optimizers.Optimizer.get_weights() was retired. Current code is experimental. Review create_cache/restore_from_cache."
            _log.error( "Cache restoration error: cached optimizer weights were not compatible with existing optimizer.%s", isTF211)
            return False
        
        return True
    
    # -------------------
    # Keras serialization    
    # -------------------

    @staticmethod
    def from_config( self, tf_config ):
        return Model( config_ID = tf_config['cache_uid'],
                      name      = tf_config['name'],
                      dtype     = tf_config['dtype'],
                      trainable = tf_config['trainable'],
                      cache_version = tf_config['cache_version']
                )
        
    def get_config( self ):
        return dict(
            name          = self.name,
            dtype         = self.type,
            trainable     = self.trainable,
            cache_uid     = self._cache_unique_id,
            cache_version = self._cache_version
            )

    # -------------------
    # protected members
    # -------------------

    def _cache_get_optimizer( self ):
        """ Utility function to obtain the config and weights of the current optimizer, and its unique ID """
        _log.verify( self._cache_ready, "Model has not yet been built. Need to execute one __call__() first")

        optimizer   = getattr( self, "optimizer", None )
        if optimizer is None:
            return None, None, ""
            
        opt_weights = optimizer.get_weights() if not getattr(self.optimizer,"get_weights",None) is None else None
        opt_config  = tf.keras.optimizers.serialize( optimizer )['config']

        if not opt_config is None and opt_weights is None:
            # tensorflow 2.11 abandons 'get_weights'
            variables   = self.optimizer.variables()        
            opt_weights = [ np.array( v ) for v in variables ]

        opt_uid   = uniqueHash( { k: opt_config[k] for k in opt_config if k != 'learning_rate' } )
        return opt_weights, opt_config, opt_uid
            
    def _cache_uid( self, opt_uid ):
        """ compute cache ID """
        return uniqueHash( [ self._cache_version, self._cache_unique_id, opt_uid ] )

# ==========================================================================
# TrainingInfo
# Information on the current training run
# ==========================================================================

class TrainingInfo(object):
    """
    Information on the current training run for user updates
    """
    
    def __init__(self, *, batch_size, epochs, num_weights):#NOQA
        self.epochs       = epochs       # epochs requested by the user. Note that the actual epochs will be reduced by the number of epochs already trained in the cached file
        self.batch_size   = batch_size   # batch size.
        self.num_weights  = num_weights  # total number of trainable weights

# ==========================================================================
# Environment
# Contains the top level data available throughout the training process
# ==========================================================================

class Environment(PrettyDict):
    """ 
    Represents the data available for the overall training loop: the model, its data, sample weights
    This means this environment can also execute a predict() on the current model for both its training and validation set.
    This is implemented in predict().
    
    Objects of this class are not serialized directly.
    
    The usual step is to create one's own, e.g. to add additional environment data
    """
    
    def __init__(self, *, model              : Model, 
                          tf_trn_data        : dict,
                          tf_val_data        : dict = None,
                          trn_sample_weights : np.ndarray = None,
                          val_sample_weights : np.ndarray = None,
                          key_loss           : str = "loss",
                          **kwargs ):
        """
        Initialize environment model
        
        Parameters
        ----------
            model : Model
                keras model derived from Model.
            tf_trn_data : dict
                Dictionary of TF data to be passed to the model during training.
                If the sample path are distributed according to some sample_weights,
                then this dictionary must contain the probabiltiy weights and key_sample_weights must
                be set to the name of this element.
            tf_val_data : dict
                Dictionary of TF data used for validation. Set to None for no validation
            trn_sample_weights : np.ndarray
                Sample weights for the training data set. None for the uniform distribution.
            val_sample_weights : np.ndarray
                Sample weights for the validation data set. None for the uniform distribution.
            key_loss : str
                Name of the primary loss vector returned from a model predict call.
                The environment will use thise to extract the current loss.
                This is used for determining the best loss (with the training data).
            **kwargs : 
                Other arguments to be passed to 'self', see PrettyDict.
        """
        self.model              = model
        self.key_loss           = str(key_loss)
        self.trn                = PrettyDict()
        self.trn.tf_data        = tf_trn_data
        self.trn.sample_weights = npCast( trn_sample_weights )
        if not self.trn.sample_weights is None:
            self.trn.sample_weights = self.trn.sample_weights[:,0] if len(self.trn.sample_weights) == 2 and self.trn.sample_weights.shape[1] == 1 else self.trn.sample_weights
            _log.verify( len(self.trn.sample_weights.shape) == 1, "'trn_sample_weights' must be a vector or of shape (N,1), but found tensor of shape %s", trn_sample_weights.shape)

        if tf_val_data is None:
            self.val = None
        else:
            self.val                = PrettyDict()
            self.val.tf_data        = tf_val_data
            self.val.sample_weights = npCast( val_sample_weights )
            _log.verify( self.trn.sample_weights is None == self.val.sample_weights is None, "'val_sample_weights' and 'trn_sample_weights' must be specified jointly, or jointly not")

            if not self.val.sample_weights is None:
                self.val.sample_weights = self.val.sample_weights[:,0] if len(self.val.sample_weights) == 2 and self.val.sample_weights.shape[1] == 1 else self.val.sample_weights
                _log.verify( len(self.trn.sample_weights.shape) == 1, "'val_sample_weights' must be a vector or of shape (N,1), but found tensor of shape %s", val_sample_weights.shape)

        if len(kwargs) > 0:
            self.update(kwargs)

    def predict(self):
        """
        Call current model on tf_data and tf_val_data to predict the latest results of the model
        
        Returns
        -------
            A PrettyDict which contains
                trn.result : numpy arrays of the training results from model(trn.tf_data)
                trn.loss   : float of the training loss for the current model               
            If val is not None:
                val.result : numpy arrays of the validation results from model(val.tf_data)
                val.loss   : float of the validation loss for the current model               
        """
        
        # training set
        pack              = PrettyDict()
        pack.trn          = PrettyDict()        
        pack.trn.results  = npCast( self.model(self.trn.tf_data) )
        _log.verify( isinstance(pack.trn.results, np.ndarray) or ( isinstance(pack.trn.results, Mapping) and self.key_loss in pack.trn.results), "The data returned from the model must either be the loss tensor, or be a dictionary with '%s' entry as specified by 'loss_key'. Model returned data type %s", self.key_loss, str(type(pack.trn.results)))

        pack.trn.loss     = pack.trn.results if isinstance(pack.trn.results, np.ndarray) else pack.trn.results[self.key_loss]
        pack.trn.loss     = pack.trn.loss[:,0] if len(pack.trn.loss.shape) == 2 and pack.trn.loss.shape[1] == 1 else pack.trn.loss
        _log.verify( len(pack.trn.loss.shape) == 1, "'loss' must be a vector or of shape (N,1). Found tensor of shape %s", pack.trn.loss.shape)
        if not self.trn.sample_weights is None:
            _log.verify( len(pack.trn.loss) == len(self.trn.sample_weights), "Invalid training sample weight vector: loss vector returned by model is of length %ld, while training sample weights are of length %ld", len(pack.trn.loss), len(self.trn.sample_weights))        
        pack.trn.loss    = np.sum( self.trn.sample_weights * pack.trn.loss ) if not self.trn.sample_weights is None else np.mean( pack.trn.loss )     

        # validation set        
        if self.val is None:
            pack.val = None
        else:
            pack.val          = PrettyDict()
            pack.val.results  = npCast( model(tf_val_data) ) 
            pack.val.loss     = pack.val.results if isinstance(pack.val.results, np.ndarray) else pack.val.results[self.key_loss]
            pack.val.loss     = pack.val.loss[:,0] if len(pack.val.loss.shape) == 2 and pack.val.loss.shape[1] == 1 else pack.val.loss
            pack.val.loss     = np.sum( self.val.sample_weights * pack.val.loss ) if not self.val.sample_weights is None else np.mean( pack.val.loss )     

        return pack
        
# ==========================================================================
# ProgressData
# Base class for relevant data to be computed during training for user
# feedback (e.g. history of losses; current state of the model)
# ==========================================================================

class ProgressData(object):
    """
    Base class for relevant data to be computed during training for user
    feedback (e.g. history of losses; current state of the model).
    
    This class is intended to be derived from, and that you overwrite on_epoch_end.
    
    For being used in Ray, this class needs to be pickle'able.
    """
    
    STOP_CONVERGED   = -1
    STOP_ABORTED     = -2
    CONTINUE         = 0
    
    def __init__(self, environment     : Environment,        # model, tf_data, etc
                       training_info   : TrainingInfo,       # total number of epochs requested etc
                       predicted_data0 : PrettyDict         
                       ):
        """
        Initialize the cachable progress data store
        ** Do not store the model or the training data into this object **
        
        Parameters
        ----------
            environment : Environment,
                provides access to various non-serializable objects in the training loop
            epochs : int
                Number of epochs to be computed.
            predicted_data0 : PrettyDict
                Result of environment.predict(). If None, this will be computed on-the-fly.
                
        """
        self.times_per_epoch = []
        self.trn_losses      = [ predicted_data0.trn.loss ]
        self.val_losses      = [ predicted_data0.val.loss ] if not predicted_data0.val is None else None
        
        # best epoch
        self.best_epoch      = -1
        self.best_weights    = environment.model.get_weights()
        self.best_loss       = predicted_data0.trn.loss
    
    @property
    def current_epoch(self):
        """ Returns the current epoch. Returns -1 if no epoch was yet recorded """
        return len(self.times_per_epoch)-1
    
    def on_epoch_end(self,  environment    : Environment,  # model, tf_data, etc
                            predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                            logs           : dict          # logs c.f. keras Callback
                        ) -> int:
        """ 
        Callback at the end of an epoch
        Return self.STOP_CONVERGED or self.STOP_ABORTED to abort training or self.CONTINUE to continue
        """
        return self.CONTINUE
    
    def on_done(self,       environment    : Environment,  # model, tf_data, etc
                            predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                        ):
        """ Called when training is finished and the model was set to the best weights """
        pass
        
    # --------------------
    # Internal
    # --------------------
            
    def _on_epoch_end(self, environment    : Environment,  # model, tf_data, etc
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                            time_epoch     : float,        # time required for 
                            logs           : dict          # logs c.f. keras Callback
                        ):
        """
        Called at the end of an epoch.
        Will store the time for the epoch in 'times_per_epoch'
        
        This function is called by the training loop.
        Do not overwrite this function; instead overwrite 'on_epoch_end'
        
        Returns
        -------
            Return self.STOP_CONVERGED or self.STOP_ABORTED to stop training,
            or self.CONTINUE to continue.
        """
        assert len(self.times_per_epoch)+1 == len(self.trn_losses), "Internal error: %ld+1 != %ld" % (len(self.times_per_epoch), len(self.trn_losses))

        predicted_data = environment.predict()

        self.times_per_epoch.append( time_epoch )
        self.trn_losses.append( predicted_data.trn.loss )
        if not self.val_losses is None:
            self.val_losses.append( predicted_data.val.loss )
        
        if self.best_loss > predicted_data.trn.loss:
            self.best_epoch   = self.current_epoch
            self.best_weights = environment.model.get_weights()
            self.besr_loss    = predicted_data.trn.loss
        
        return self.on_epoch_end( environment=environment, predicted_data=predicted_data, training_info=training_info, logs=logs )
        
    def _on_done(self,      environment    : Environment,  # model, tf_data, etc
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                        ):
        predicted_data = environment.predict()
        self.on_done(  environment=environment, predicted_data=predicted_data, training_info=training_info )
        
# ==========================================================================
# Callback
# This is called during training to handle caching and user updates
# ==========================================================================

class Callback(tf.keras.callbacks.Callback):
    """
    Manages training of our model    
    -- Keeps track of training data in TrainingProgressData including best fit
    -- Implements caching
    -- Implements dyanmic visual updates
    """
    
    STOP_ABORTED     = ProgressData.STOP_ABORTED
    STOP_CONVERGED   = ProgressData.STOP_CONVERGED
    CONTINUE         = ProgressData.CONTINUE
    STOP_INTERRUPTED = -10
    FINISHED_EPOCHS  = 1
    ALREADY_TRAINED  = 2
    
    def __init__(self, *, environment    : Environment,
                          training_info  : TrainingInfo,
                          create_progress: ProgressData,
                          config         : Config = Config(),
                          verbose        : Context = Context() ):
        """
        Initialize the call back
        The object will attempt to restore a previous training state from disk if found.
        
        Parameters
        ----------
            model_cachable
                Model derived from Model.
            epochs : int
                Total number of epochs for this training. If the cached object has been trained beyond that point, no further training will commence.
            default_cache_directory : str
                Default caching directory for 
        """
        
        tf.keras.callbacks.Callback.__init__(self)
        
        model                 = environment.model
        _log.verify( isinstance(model, Model), "'model' must be derived from 'Model'")
        _log.verify( model.is_caching_ready, "'model' must have been called at least once")
        _log.verify( model.has_optimizer, "'model' must have been compiled: not optimizer was found")

        # basics
        self.environment      = environment
        self.training_info    = training_info
        self.progress_data    = create_progress(environment=environment, training_info=training_info, config=config.progress )
        self.cache_last_epoch = None
        self.verbose          = verbose
        self.time_start       = time.time()
        self.stop_reason      = self.CONTINUE
        _log.verify( self.training_info.epochs > 0, "'epochs' must be positive. Found %ld", self.training_info.epochs )

        # caching
        def_directory_name    = model.cache_def_directory_name
        self.cache_mode       = config.caching("mode", CacheMode.ON, CacheMode.MODES, "Caching strategy: %s" % CacheMode.HELP)
        self.cache_dir        = config.caching("directory", "./.cache/" + def_directory_name, str, "Caching directory")
        self.cache_freq       = config.caching("epoch_freq", 10, Int>0, "How often to cache results, in number of epochs")
        cache_file_name       = config.caching("file_name", "", str, "Allows overwriting the filename for debugging an explicit cached state")
        config.done()
            
        self.cache_mode       = CacheMode( self.cache_mode )
        self.cache_dir        = SubDir(self.cache_dir, "!")
        self.cache_id         = model.cache_uid
        self.cache_file       = uniqueFileName48( self.cache_id ) if len(self.cache_id) > 48 else self.cache_id
        self.cache_file       = self.cache_file if cache_file_name == "" else cache_file_name
        self.full_cache_file  = self.cache_dir.fullKeyName( self.cache_file )
        self.cache_data       = None
        
        # restore cache
        # this might overwrite self.progress_data
        self.cache_restore    = None
        if not self.cache_mode.is_off:
            verbose.report(0, "Caching enabled @ '%s'" % self.full_cache_file)
            if self.cache_mode.delete:
                self.cache_dir.delete( self.cache_file )    
            elif self.cache_mode.read:
                # restore cache                
                cache = self.cache_dir.read( self.cache_file )
                if not cache is None:
                    # load everything except the gym 
                    # restore gym
                    self.cache_data = cache
                    if not model.cache_restore( cache['model'], initial=True ):
                        if self.cache_mode.del_incomp:
                            self.cache_dir.delete( self.cache_file ) 
                            verbose.report(1, "Cache consistency error: could not write weights from cache '%s' to current model. This is most likely because the model architecture changed.\n"\
                                              "The file was deleted because caching mode was '%s'",\
                                              self.full_cache_file, self.cache_mode )
                        else:
                            verbose.report(1, "Cache consistency error: could not write weights from cache '%s' to current model. This is most likely because the model architecture changed.\n"\
                                              "Set caching model to '%s' to rebuild caches which are not compatible with the current code base. Use caching model '%s' to turn caching off.",\
                                              self.full_cache_file, CacheMode.ON, CacheMode.OFF )
                    else:
                        self.progress_data = cache['progress_data']
                        _log.verify( self.progress_data.current_epoch >= 0, "Error: object restored from cache had epoch set to %ld", self.progress_data.current_epoch )
                        self.cache_last_epoch = self.progress_data.current_epoch
                        verbose.report(1, "Cache successfully loaded. Current epoch: %ld" % (self.progress_data.current_epoch+1) )

        # initialize timing
        if self.progress_data.current_epoch+1 >= self.training_info.epochs:
            verbose.report(0, \
                   "Nothing to do: cached model loaded from %s was trained for %ld epochs; you have asked to train for %ld epochs.\n"\
                   "If you want to force training: raise number of epochs or turn off caching.", \
                   self.full_cache_file, self.progress_data.current_epoch+1, self.epochs )
            self.stop_reason = self.ALREADY_TRAINED
        self.time0 = time.time()

    def write_cache(self):
        """ Write cache to disk """
        if not self.cache_data is None:
            return # calibration was never invoked. Keep existing cache
        
        cache = { 'model':         self.environment.model.cache_create(),
                  'progress_data': self.progress_data
                }            
        self.cache_dir.write( self.cache_file, cache )
        self.cache_last_epoch = self.progress_data.current_epoch
        
    @property
    def is_done(self):
        """ Checks whether training has finished. This can happen at inception if a cache is restored which was trained for as many epochs as requested """
        return self.progress_data.current_epoch+1 >= self.training_info.epochs
    
    @property
    def current_epoch(self):
        """ Returns the current epoch. -1 if no epoch was run """
        return self.progress_data.current_epoch
    
    @property
    def epochs(self):
        return self.training_info.epochs

    def on_epoch_begin( self, loop_epoch, logs = None ):
        pass
    
    def on_epoch_end( self, loop_epoch, logs = None ):
        """
        Called when an epoch ends
        Handle plotting, and caching
        Note that 'loop_epoch' is the epoch of the current training run. If the state was recovered from a cache, it won't be the logical epoch
        """

        if not self.cache_data is None:
            model = self.environment.model
            if not model.cache_restore( self.cache_data['model'], initial=False ):
                if self.cache_mode.del_incomp:
                    self.cache_dir.delete( self.cache_file ) 
                    self.verbose.report(1, "Cache consistency error: could not write weights from cache '%s' to current model. This is most likely because the model architecture changed.\n"\
                                      "The file was deleted because caching mode was '%s'",\
                                      self.full_cache_file, self.cache_mode )
                else:
                    self.verbose.report(1, "Cache consistency error: could not write weights from cache '%s' to current model. This is most likely because the model architecture changed.\n"\
                                      "Set caching model to '%s' to rebuild caches which are not compatible with the current code base. Use caching model '%s' to turn caching off.",\
                                      self.full_cache_file, CacheMode.ON, CacheMode.OFF )
            else:
                self.progress_data = self.cache_data['progress_data']
                _log.verify( self.progress_data.current_epoch >= 0, "Error: object restored from cache had epoch set to %ld", self.progress_data.current_epoch )
                self.cache_last_epoch = self.progress_data.current_epoch
                self.verbose.report(1, "Optimizer state successfully loaded from cache. Current epoch: %ld" % (self.progress_data.current_epoch+1) )
            self.cache_data = None
            self.time_start = time.time()
            return
                    
        time_now = time.time()
        _current = self.progress_data.current_epoch
        r = self.progress_data._on_epoch_end( environment   = self.environment,
                                              training_info = self.training_info,
                                              time_epoch    = time_now - self.time_start,
                                              logs          = logs )
        assert self.progress_data.current_epoch >= 0, "Internal error: progress_data must update its epoch count"
        assert self.progress_data.current_epoch > _current, "Internal error: progress_data must update its epoch count"

        # allow calling progress data to abort training
        if r in [ProgressData.STOP_ABORTED, ProgressData.STOP_CONVERGED]:
            self.write_cache()
            self.stop_reason         = r
            self.model.stop_training = True

        self.time_start = time_now
        
        if self.current_epoch % self.cache_freq == 0 and self.cache_mode.write and (self.cache_last_epoch is None or self.current_epoch > self.cache_last_epoch):
            self.write_cache()
            
    def finalize( self ):
        """
        Close training. Call this even if training was aborted
        -- Cache the current state
        -- Apply best weight
        """
        # cache current state /before/ we reset gym to its best weights
        # this way we can continue to train from where we left it
        cached_msg = ""
        if self.progress_data.current_epoch >= 0 and self.cache_mode.write:
            self.write_cache()
            cached_msg = " State of training until epoch %ld cached into %s\n" % (self.cache_last_epoch+1, self.full_cache_file)
            
        status = ""
        if self.stop_reason == self.STOP_ABORTED:
            status = "Training aborted"
        elif self.stop_reason == self.STOP_CONVERGED:
            status = "Desired convergence achieved"
        elif self.stop_reason == self.STOP_INTERRUPTED:
            status = "User abort"
        elif self.stop_reason == self.FINISHED_EPOCHS:
            status = "Trained %ld epochs" % self.epochs
        elif self.stop_reason == self.ALREADY_TRAINED:
            status = "Model was already trained for at least %ld epochs" % self.epochs
        else:
            _log.throw("Unknown stopping reason %ld", self.stop_reason)            

        # restore best weights
        # We do this *after* we stored the last cache
        self.environment.model.set_weights( self.progress_data.best_weights )

        self.progress_data._on_done(  environment   = self.environment,
                                      training_info = self.training_info )
        """ Called when training is finished and the model was set to the best weights """
        pass
        
        self.verbose.write( "Status: %(status)s.\n"\
                           "Weights set to best epoch: %(best_epoch)ld\n"\
                           "%(cached_msg)s Time: %(time)s",\
                           status=status, 
                           best_epoch=self.progress_data.best_epoch+1,
                           cached_msg=cached_msg,
                           time=fmt_now())

# ==========================================================================
# Main training loop
# ==========================================================================

@tf.function
def default_loss( y_true,y_pred ):     
    """ Default loss: ignore y_true """
    return y_pred

def train(   environment    : Environment,
             create_progress: type = ProgressData,
             config         : Config = Config(),
             verbose        : Context = Context() ):
    """
    Main training loop
    
    Parameters
    ----------
        environment : Environment
            Contains the model, and the training and validation data sets. Also contains sample weights
            You can provide a derived class if you wish to pass on further information to progess_data.on_epoch_end
            Alternatively, you can pass a dictionary with the required elements to construct an Environment object
        progress_data : ProgressData
            Main callback: the function on_epoch_end() is called at the end of each epoch.
            This function is then intended to compute all other summary statistics required for user feedback doing training.
            The object needs to be pickle'abel if it is intended to be used a multiprocessing environment such as Ray
        config : Config
            Standard config
        verbose :
            Controls level of output.

    Returns
    -------
        A PrettyDict which contains, computed at the best weights:
            model         : trained model, set to best weights (according to training data)
            progress_data : progress data, e.g. a version of ProgressData which contains at the very least the time series of losses, and the best weights
            trn.result    : numpy arrays of the training results from model(trn.tf_data)
            trn.loss      : float of the training loss for the current model               
        If val is not None:
            val.result    : numpy arrays of the validation results from model(val.tf_data)
            val.loss      : float of the validation loss for the current model               
    """
    verbose.write("Training loop starting")
    t0  = time.time()
    
    # how much to print
    debug_numerics   = config.debug("check_numerics", False, bool, "Whether to check TF numerics.")
    
    # training parameters    
    batch_size       = config.train("batch_size",  None, help="Batch size")
    epochs           = config.train("epochs",      100, Int>0, help="Epochs")
    run_eagerly      = config.train("run_eagerly", False, help="Keras model run_eagerly. Turn to True for debugging. This slows down training. Use None for default.")
    learning_rate    = config.train("learing_rate", None, help="Manually set the learning rate of the optimizer")
    tf_verbose       = config.train("tf_verbose", 0, Int>=0, "Verbosity for TensorFlow fit()")
    optimzier        = create_optimizer(config.train)
    
    # tensorboard: have not been able to use it .. good luck.
    tboard_log_dir   = config.train.tensor_board(   "log_dir", "", str, "Specify tensor board log directory. See https://www.tensorflow.org/guide/profiler")
    tboard_freq      = config.train.tensor_board(   "hist_freq", 1, Int>0, "Specify tensor board log frequency. See https://www.tensorflow.org/guide/profiler") 
    tboard_prf_batch = config.train.tensor_board(   "profile_batch", 0, help="Batch used for profiling. Set to non-zero to activate profiling. See https://www.tensorflow.org/guide/profiler") 

    # compile
    # -------

    if isinstance(environment.model, type):
        environment.model = model( config = config.model, name = model.__name__, dtype=dtype )
    model          = environment.model    
    t0             = time.time()
    pack0          = environment.predict()
    verbose.write("Model evaluated with current weights. Model has %ld weights." % model.num_trainable_weights)
    
    model.compile(  optimizer        = optimzier, 
                    loss             = { environment.key_loss : default_loss },
                    weighted_metrics = { environment.key_loss : default_loss },
                    run_eagerly      = run_eagerly)
    if not learning_rate is None:
        gym.optimizer.lr = float( learning_rate )
    
    # prepare tracking
    # ----------------
    
    training_info = TrainingInfo( batch_size     = batch_size,
                                  epochs         = epochs,
                                  num_weights    = model.num_trainable_weights)        
    callback      = Callback(     environment    = environment,
                                  training_info  = training_info,
                                  create_progress= create_progress,
                                  config         = config,
                                  verbose        = verbose.sub(1) )
    config.done()

    # train
    # -----
    
    if debug_numerics:
        tf.debugging.enable_check_numerics()
        verbose.report(1, "Enabled automated checking for numerical errors. This will slow down training. Use config.debug.check_numerics = False to turn this off")
    else:
        tf.debugging.disable_check_numerics()
    
    if not callback.is_done:
        assert epochs > (callback.current_epoch+1), "Internal error. callback.is_done failed"
        # tensorboard
        # See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html

        tboard = None
        if tboard_log_dir != "":
            t0             = time.time()
            tboard_log_dir = SubDir(tboard_log_dir).path
            tboard         = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, histogram_freq=tboard_freq, profile_batch=tboard_prf_batch )
            verbose.report(1,"TensorBoard log directory set to '%s'. Took %s" % (tboard_log_dir, fmt_seconds(time.time()-t0)))

        def find_sample_size( x ):
            if isinstance(x, tf.Tensor):
                assert int(x.shape[0])>0, x.shape
                return int(x.shape[0])
            if isinstance(x, dict):
                for x in x.values():
                    l = find_sample_size(x)
                    if l>0:
                        return l
            else:
                assert isinstance(x, list), "Cannot find sample size. Type %s" % str(type(x))
                for x in x:
                    l = find_sample_size(x)
                    if l>0:
                        return l
            return 0
        nSamples = find_sample_size(environment.trn.tf_data)

        try:
            model.fit(      x              = environment.trn.tf_data,
                            y              = tf.zeros((nSamples,), dtype=model.dtype),
                            batch_size     = batch_size,
                            sample_weight  = environment.trn.sample_weights * float(len(environment.trn.sample_weights)) if not environment.trn.sample_weights is None else None,  # sample_weights are poorly handled in TF
                            epochs         = epochs - (callback.current_epoch+1),
                            callbacks      = callback if tboard is None else [ callback, tboard ],
                            verbose        = tf_verbose )
            callback.stop_reason = Callback.FINISHED_EPOCHS            
        except KeyboardInterrupt:
            callback.stop_reason = Callback.STOP_INTERRUPTED

    callback.finalize()
    verbose.report(0, "Training completed. Total training took %s", fmt_seconds(time.time()-t0))
    result = environment.predict()
    result.progress_data = callback.progress_data
    result.model = model
    return result




