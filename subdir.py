# -*- coding: utf-8 -*-
"""
subdir
Simple class to keep track of directory sturctures and for automated caching on disk
Hans Buehler 2020

WARNING
This is under development. I have not figured out how to test file i/o on GitHub
"""

from cdxbasics.logger import Logger
from cdxbasics.util import uniqueHash
_log = Logger(__file__)

import os
import os.path
from functools import wraps
from hashlib import shake_128
import pickle
import tempfile
import shutil
import datetime
from collections.abc import Collection, Mapping

def _remove_trailing( path ):
    if len(path) > 0:
        if path[-1] in ['/' or '\\']:
            return _remove_trailing(path[:-1])
    return path

class SubDir(object):
    """
    SubDir implements a transparent interface for storing data in files, with a common extension.
    The generic pattern is:
        
        1) create a root 'parentDir':
            Absolute:                      parentDir = SubDir("C:/temp/root") 
            In system temp directory:      parentDir = SubDir("!/root")
            In user directory:             parentDir = SubDir("~/root")
            Relative to current directory: parentDir = SubDir("./root")

        2) Use SubDirs to transparently create hierachies of stored data:
           assume f() will want to store some data:
               
               def f(parentDir, ...):
                
                   subDir = parentDir('subdir')    <-- note that the call () operator is overloaded: if a second argument is provided, the directory will try to read the respective file.
                   or
                   subDir = SubDir('subdir', parentDir)   
                    :
                    :
            Write data:
                        
                   subDir['item1'] = item1       <-- dictionary style
                   subDir.item2 = item2          <-- member style
                   subDir.write('item3',item3)   <-- explicit

            Note that write() can write to multiple files at the same time.
                   
        3) Reading is similar

                def readF(parentDir,...):
                    
                    subDir = parentDir('subdir')
                    
                    item = subDir('item', 'i1')     <-- returns 'i1' if not found.
                    item = subdir.read('item')      <-- returns None if not found
                    item = subdir.read('item','i2') <-- returns 'i2' if not found
                    item = subDir['item']           <-- returns None if not found unless subDir._raiseOnError is True in which case a KeyError is thrown
                    item = subDir.item              <-- returns None if not found unless subDir._raiseOnError is True in which case a KeyError is thrown
                    
        4) Treating data like dictionaries
    
                def scanF(parentDir,...)
                
                    subDir = parentDir('f')
                    
                    for item in subDir:
                        data = subDir[item]
                        
            Delete items:
                
                    del subDir['item']             <-- silently fails if 'item' not exist unless subDir._raiseOnError is True in which case a KeyError is thrown
                    del subDir.item                <-- silently fails if 'item' not exist unless subDir._raiseOnError is True in which case a KeyError is thrown
                    subDir.delete('item')          <-- silently fails if 'item' not exist unless subDir._raiseOnError is True in which case a KeyError is thrown    
                    subDir.delete('item', True)    <-- throw a KeyError if 'item' does not exit

        5) Cleaning up
    
                parentDir.deleteAllContent()       <-- deletes all files and sub directories.
                        
        Several other operations are supported; see help()
        
        Hans Buehler May 2020
    """
    class __RETURN_SUB_DIRECTORY(object):
        pass
  
    DEFAULT_EXT = "pck"
    DEFAULT_RAISE_ON_ERROR = False
    RETURN_SUB_DIRECTORY = __RETURN_SUB_DIRECTORY     # comparison between classes is unique accross Python name space
    
    def __init__(self, name : str, parent = None, *, ext : str = None, raiseOnError : bool = None, eraseEverything : bool = False ):
        """ 
        Creates a sub directory which contains pickle files with a common extension.
        
        Absolute directories        
            sd  = SubDir("!/subdir")           - relative to system temp directory
            sd  = SubDir("~/subdir")           - relative to user home directory
            sd  = SubDir("./subdir")           - relative to current working directory (explicit)
            sd  = SubDir("subdir")             - relative to current working directory (implicit)
            sd  = SubDir("/tmp/subdir")        - absolute path (linux)
            sd  = SubDir("C:/temp/subdir")     - absolute path (windows)
        Short-cut
            sd  = SubDir("")                   - current working directory

        It is often desired that the user specifies a sub-directory name under some common parent directory.
        You can create sub directories if you provide a 'parent' directory:
            sd2 = SubDir("subdir2", parent=sd) - relative to other sub directory
            sd2 = sd("subdir2")                - using call operator
        Works with strings, too:
            sd2 = SubDir("subdir2", parent="~/my_config") - relative to ~/my_config
            
        All files managed by SubDir will have the same extension.
        The extension can be specified with 'ext', or as part of the directory string:
            
            sd  = SubDir("~/subdir;*.bin")      - set extension to 'bin'
        
        COPY CONSTRUCTION
        The function also allows copy construction, and constrution from a repr() string.
        
        HANDLING KEYS
        SubDirs allows reading data using the item and attribute notation, i.e. we may
            sd = SubDir("~/subdir")
            x  = sd.x
            y  = sd['y']
        The default handling is to return None /not/ to throw an error.
        This can be turned off using raiseOnError = True.
        
        NONE OBJECTS
        It is possible to set the directory name to 'None'. In this case the directory will behave as if:
            No files exist
            Writing fails with a EOFError.
        
                
        Parameters
        ----------
            name          - Name of the directory.
                               '.' for current directory
                               '~' for home directory
                               '!' for system default temp directory
                            May contain a formatting string for defining 'ext' on the fly:
                                Use "!/test;*.bin" to specify 'test' in the system temp directory as root directory with extension 'bin'
                            Can be set to None, see above.                            
            parent         - Parent directory. If provided, will also set defaults for 'ext' and 'raiseOnError'
            ext            - standard file extenson for data files. All files will share the same extension.
                             If None, use the parent extension, or if that is not specified DEFAULT_EXT (pck)
                             Set to "" to turn off managing extensions.
            raiseOnError   - if False, the following operations will be silent if 'key' does not exist:
                                subdir.key        - returns None
                                subdir['key']     - returns None
                                del subdir.key    - does not do anything, silently
                                del subdir['key'] - does not do anything, silently
                             if True, then the above will throw KeyErrors.
                             if None, copy default from 'parent' or set to DEFAULT_RAISE_ON_ERROR (false)
            eraseEverything - delete all contents in the newly defined subdir
        """
        # copy constructor support
        if isinstance(name, SubDir):
            assert parent is None, "Internal error: copy construction does not accept additional keywords"
            self._path = name._path
            self._ext = name._ext if ext is None else ext
            self._raiseOnError = name._raiseOnError if raiseOnError is None else raiseOnError
            return
        
        # reconstruction from repr()
        if isinstance(name, Mapping):
            assert parent is None, "Internal error: dictionary construction does not accept additional keywords"
            self._path = name['_path']
            self._ext = name['_ext'] if ext is None else ext
            self._raiseOnError = name['_raiseOnError'] if raiseOnError is None else raiseOnError
            return
        
        # parent
        if isinstance(parent, str):
            parent = SubDir(parent, ext=ext, raiseOnError=raiseOnError)
        _log.verify( parent is None or isinstance(parent, SubDir), "'parent' must be SubDir or None. Found object of type %s", type(parent))

        # operational flags
        self._raiseOnError = raiseOnError if not raiseOnError is None else (parent._raiseOnError if not parent is None else SubDir.DEFAULT_RAISE_ON_ERROR)
        _name  = name if not name is None else "(none)"
        
        # extension
        if not name is None:
            _log.verify( isinstance(name, str), "'name' must be string. Found object of type %s", type(name))
            name   = name.replace('\\','/') 

            # extract extension information
            ext_i = name.find(";*.")
            if ext_i >= 0:
                _ext = name[ext_i+3:]
                _log.verify( ext is None or ext == _ext, "Canot specify an extension both in the name string ('%s') and as 'ext' ('%s')", _name, ext)
                ext  = _ext
                name = name[:ext_i]        
        if ext is None:
            self._ext = ("." + SubDir.DEFAULT_EXT) if parent is None else parent._ext
        else:
            _log.verify( isinstance(ext,str), "Extension 'ext' must be a string. Found type %s", type(ext))
            if len(ext) == 0:
                self._ext = ""
            else:
                _log.verify( not ext in ['.','/','\\'], "Extension 'ext' cannot be '%s'", ext )
                sub, _ = os.path.split(ext)
                _log.verify( len(sub) == 0, "Extension 'ext' '%s' contains directory information", ext)
                self._ext = ("." + ext) if ext[:1] != '.' else ext

        # name
        if name is None:
            if not parent is None and not parent._path is None:
                name = parent._path[:-1]
        else:
            # expand name
            name = _remove_trailing(name)
            if name == "" and parent is None:
                name = "."
            if name[:1] in ['.', '!', '~']:
                _log.verify( len(name) == 1 or name[1] == '/', "If 'name' starts with '%s', then the second character must be '/' (or '\\' on windows). Found 'name' set to '%s'", name[:1], _name)
                if name[0] == '!':
                    name = SubDir.tempDir()[:-1] + name[1:] 
                elif name[0] == ".":
                    name = SubDir.workingDir()[:-1] + name[1:]
                else:
                    assert name[0] == "~"
                    name = SubDir.userDir()[:-1] + name[1:]
            elif not parent is None:
                # path relative to 'parent'
                name = (parent._path + name) if not parent.is_none else name

        if name is None:
            self._path = None
        else:
            # expand path
            self._path = os.path.abspath(name) + '/'
            self._path = self._path.replace('\\','/') 

            # create directory
            if not os.path.exists( self._path[:-1] ):
                os.makedirs( self._path[:-1] )
            else:
                _log.verify( os.path.isdir(self._path[:-1]), "Cannot use sub directory %s: object exists but is not a directory", self._path[:-1] )
                # erase all content if requested
                if eraseEverything:
                    self.eraseEverything(keepDirectory = True)
                                                
    # -- self description --
        
    def __str__(self) -> str: # NOQA
        if self._path is None: return "(none)"
        return self._path if len(self._ext) == 0 else self._path + ";*" + self._ext
    
    def __repr__(self) -> str: # NOQA
        return repr({'path':self._path, 'ext':self._ext, 'raiseOnError':self._raiseOnError})
    
    def __eq__(self, other) -> bool: # NOQA
        """ Tests equality between to SubDirs, or between a SubDir and a directory """
        if isinstance(other,str):
            return self._path == other
        _log.verify( isinstance(other,SubDir), "Cannot compare SubDir to object of type '%s'", type(other))
        return self._path == other._path and self._ext == other._ext and self._raiseOnError == other._raiseOnError

    def __bool__(self) -> bool:
        """ Returns True if 'self' is set, or False if 'self' is a None directory """
        return not self.is_none
    
    def __hash__(self) -> str: #NOQA
        return hash( (self._path, self._ext, self._raiseOnError) )

    @property
    def is_none(self) -> bool:
        """ Whether this object is 'None' or not """
        return self._path is None
    
    @property
    def path(self) -> str:
        """ Return current path, including trailing '/' """
        return self._path

    def fullKeyName(self, key : str) -> str:
        """
        Returns fully qualified key name
        Note this function is not robustified against 'key' containing directory features
        """
        if self._path is None:
            return None
        if len(self._ext) > 0 and key[-len(self._ext):] != self._ext:
                 
            return self._path + key + self._ext
        return self._path + key

    @staticmethod
    def tempDir() -> str:
        """
        Return system temp directory. Short cut to tempfile.gettempdir()
        Result contains trailing '/'
        """
        d = tempfile.gettempdir()
        _log.verify( len(d) == 0 or not (d[-1] == '/' or d[-1] == '\\'), "*** Internal error 13123212-1: %s", d)
        return d + "/"
    
    @staticmethod
    def workingDir() -> str:
        """
        Return current working directory. Short cut for os.getcwd() 
        Result contains trailing '/'
        """
        d = os.getcwd()
        _log.verify( len(d) == 0 or not (d[-1] == '/' or d[-1] == '\\'), "*** Internal error 13123212-2: %s", d)
        return d + "/"
    
    @staticmethod
    def userDir() -> str:
        """
        Return current working directory. Short cut for os.path.expanduser('~')
        Result contains trailing '/'
        """
        d = os.path.expanduser('~')
        _log.verify( len(d) == 0 or not (d[-1] == '/' or d[-1] == '\\'), "*** Internal error 13123212-3: %s", d)
        return d + "/"
    
    # -- read --
    
    def _read( self, reader, key : str, default, raiseOnError : bool ):
        """
        Utility function for read() and readLine()
        
        Parameters
        ----------
            reader( key, fullFileName, default )
                A function which is called to read the file once the correct directory is identified
                key : key (for error messages, might include '/')
                fullFileName : full file name
                default value
            key : str or list
                str: fully qualified key
                list: list of fully qualified names
            default :
                default value. None is a valid default value
                list : list of defaults for a list of keys
            raiseOnError : bool
                If True, and the file does not exist, throw exception
        """
        # vector version
        if not isinstance(key,str):
            _log.verify( isinstance(key, Collection), "'key' must be a string, or an interable object. Found type %s", type(key))
            l = len(key)
            if default is None or isinstance(default,str) or getattr(default,"__iter__",None) is None:
                default = [ default ] * l
            else:
                _log.verify( len(default) == l, "'default' must have same lengths as 'key', found %ld and %ld", len(default), l )
            return [ self._read(reader=reader,key=k,default=d,raiseOnError=raiseOnError) for k, d in zip(key,default) ]

        # deleted directory?
        if self._path is None:
            _log.verify( not raiseOnError, "Trying to read '%s' from an empty directory object", key)
            return default
        
        # single key
        _log.verify(len(key) > 0, "'key' missing (the filename)" )
        sub, key = os.path.split(key)
        if len(sub) > 0:
            return SubDir(self,sub)._read(reader,key,default)
        _log.verify(len(key) > 0, "'key' %s indicates a directory, not a file", key)

        # does file exit?
        fullFileName = self.fullKeyName(key)
        if not os.path.exists(fullFileName):
            if raiseOnError:
                raise KeyError(key)
            return default
        _log.verify( os.path.isfile(fullFileName), "Cannot read %s: object exists, but is not a file (full path %s)", key, fullFileName )

        # read content        
        # delete existing files upon read error
        try:
            return reader( key, fullFileName, default )
        except EOFError:
            try:
                os.remove(fullFileName)
                _log.warning("Cannot read %s; file deleted (full path %s)",key,fullFileName)
            except Exception:
                _log.warning("Cannot read %s; attempt to delete file failed (full path %s)",key,fullFileName)
        if raiseOnError:
            raise KeyError(key)
        return default
    
    def read( self, key, default = None, raiseOnError : bool = False ):
        """
        Read pickled data from 'key' if the file exists, or return 'default'
        -- Supports 'key' containing directories
        -- Supports 'key' being iterable.
           In this case any any iterable 'default' except strings are considered accordingly.
           In order to have a unit default which is an iterable, you will have to wrap it in another iterable, e.g.
           E.g.:
              keys = ['file1', 'file2']

              sd.read( keys )
              --> works, both are using default None

              sd.read( keys, 1 )
              --> works, both are using default '1'

              sd.read( keys, [1,2] )
              --> works, defaults 1 and 2, respectively
              
              sd.read( keys, [1] )   
              --> produces error as len(keys) != len(default)    
              
            Strings are iterable but are treated as single value.
            Therefore
                sd.read( keys, '12' )
            means the default value '12' is used for both files.
            Use
                sd.read( keys, ['1','2'] )
            in case the intention was using '1' and '2', respectively.
              
        Returns the read object, or a list of objects if 'key' was iterable.
        If the current directory is 'None', then behaviour is as if the file did not exist.
        """
        def reader( key, fullFileName, default ):
            with open(fullFileName,"rb") as f:
                return pickle.load(f)
        return self._read( reader=reader, key=key, default=default, raiseOnError=raiseOnError )
    
    get = read

    def readString( self, key, default = None, raiseOnError = False ):
        """
        Reads text from 'key' or returns 'default'. Removes trailing EOLs
        -- Supports 'key' containing directories#
        -- Supports 'key' being iterable. In this case any 'default' can be a list, too.
        
        Returns the read string, or a list of strings if 'key' was iterable.
        If the current directory is 'None', then behaviour is as if the file did not exist.

        See additional comments for read()
        """        
        def reader( key, fullFileName, default ):
            with open(fullFileName,"r") as f:
                line = f.readline()
                if len(line) > 0 and line[-1] == '\n':
                    line = line[:-1]
                return line
        return self._read( reader=reader, key=key, default=default, raiseOnError=raiseOnError )
        
    # -- write --

    def _write( self, writer, key, obj ):
        """ Utility function for write() and writeLine() """
        if self._path is None:
            raise EOFError("Cannot write to '%s': current directory is not specified" % key)
        
        # vector version
        if not isinstance(key,str):
            _log.verify( isinstance(key, Collection), "'key' must be a string or an interable object. Found type %s", type(key))
            l = len(key)
            if obj is None or isinstance(obj,str) or not isinstance(obj, Collection):
                obj = [ obj ] * l
            else:
                _log.verify( len(obj) == l, "'obj' must have same lengths as 'key', found %ld and %ld", len(obj), l )
            for (k,o) in zip(key,obj):
                self._write( writer, k, o )
            return

        # single key
        _log.verify(len(key) > 0, "'key is empty (the filename)" )
        sub, key = os.path.split(key)
        _log.verify(len(key) > 0, "'key '%s' refers to a directory, not a file", key)
        if len(sub) > 0:
            return SubDir(self,sub)._write(writer,key,obj)
        fullFileName = self.fullKeyName(key)
        writer( key, fullFileName, obj )
    
    def write( self, key, obj ):
        """
        pickles 'obj' into key.
        -- Supports 'key' containing directories
        -- Supports 'key' being a list.
           In this case, if obj is an iterable it is considered the list of values for the elements of 'keys'
           If 'obj' is not iterable, it will be written into all 'key's
           
              keys = ['file1', 'file2']

              sd.write( keys, 1 )
              --> works, writes '1' in both files.

              sd.read( keys, [1,2] )
              --> works, writes 1 and 2, respectively

              sd.read( keys, "12" )
              --> works, writes '12' in both files
              
              sd.write( keys, [1] )   
              --> produces error as len(keys) != len(obj)                 
           
        If the current directory is 'None', then the function throws an EOFError exception
        """
        def writer( key, fullFileName, obj ):
            with open(fullFileName,"wb") as f:
                pickle.dump(obj,f,-1)
        self._write( writer=writer, key=key, obj=obj )
        
    set = write

    def writeString( self, key, line ):
        """
        writes 'line' into key. A trailing EOL will not be read back
        -- Supports 'key' containing directories
        -- Supports 'key' being a list.
           In this case, line can either be the same value for all key's or a list, too.
           
        If the current directory is 'None', then the function throws an EOFError exception
        See additional comments for write()
        """
        if len(line) == 0 or line[-1] != '\n':
            line += '\n'
        def writer( key, fullFileName, obj ):            
            with open(fullFileName,"w") as f:
                f.write(obj)
        self._write( writer=writer, key=key, obj=line )
               
    # -- iterate --
    
    def keys(self) -> list:
        """
        Returns a list of keys in this subdirectory with the correct extension.
        Note that the keys do not include the extension themselves.
        
        In other words, if the extension is ".pck", and the files are "file1.pck", "file2.pck", "file3.bin"
        then this function will return [ "file1", "file2" ]

        This function ignores directories.
        
        If self is None, then this function returns an empty list.
        """
        if self._path is None:
            return []        
        ext_l = len(self._ext)
        keys = []
        with os.scandir(self._path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if ext_l > 0:
                    if len(entry.name) <= ext_l or entry.name[-ext_l:] != self._ext:
                        continue
                    keys.append( entry.name[:-ext_l] )
                else:
                    keys.append( entry.name )
        return keys
    
    def subDirs(self) -> list:
        """
        Returns a list of all sub directories
        If self is None, then this function returns an empty list.
        """
        # do not do anything if the object was deleted
        if self._path is None:
            return []
        _log.verify( not self._path is None, "Object has been deleted")
        subdirs = []
        with os.scandir(self._path[:-1]) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                subdirs.append( entry.name )
        return subdirs
    
    # -- delete --
    
    def delete( self, key, raiseOnError = None ):
        """
        Deletes 'key'; 'key' might be a list.
        
        Parameters
        ----------
            key : filename, or list of filenames
            raiseOnError : if False, do not throw KeyError if file does not exist. If None, use subdir's default.
        """
        raiseOnError = raiseOnError if not raiseOnError is None else self._raiseOnError
        # do not do anything if the object was deleted
        if self._path is None:
            if raiseOnError: raise EOFError("Cannot delete '%s': current directory not specified" % key)
            return
        # vector version
        if not isinstance(key,str):
            _log.verify( isinstance(key, Collection), "'key' must be a string or an interable object. Found type %s", type(key))
            for k in key:
                self.delete(k, raiseOnError=raiseOnError)
            return
        # single key
        _log.verify(len(key) > 0, "'key' is empty" )
        sub, key2 = os.path.split(key)
        _log.verify(len(key2) > 0, "'key' %s indicates a directory, not a file", key)
        fullFileName = self.fullKeyName(key)
        if not os.path.exists(fullFileName):
            if raiseOnError:
                raise KeyError(key)
        else:
            os.remove(fullFileName)
        
    def deleteAllKeys( self, raiseOnError = None ):
        """ Deletes all valid keys in this sub directory """
        # do not do anything if the object was deleted
        if self._path is None:
            if raiseOnError: raise EOFError("Cannot delete all files: current directory not specified")
            return
        self.delete( self.keys(), raiseOnError=raiseOnError )
            
    def deleteAllContent( self, deleteSelf = False, raiseOnError = None ):
        """
        Deletes all valid keys and subdirectories in this sub directory.
        Does not delete files with other extensions.
        Use eraseEverything() if the aim is to delete everything.
        """
        # do not do anything if the object was deleted
        if self._path is None:
            if raiseOnError: raise EOFError("Cannot delete all contents: current directory not specified")
            return
        # delete sub directories       
        subdirs = self.subDirs();
        for subdir in subdirs:
            SubDir(root=self,subdir=subdir).deleteAllContent( deleteSelf=True, raiseOnError=raiseOnError )
        # delete keys
        self.deleteAllKeys( raiseOnError=raiseOnError )
        # delete myself    
        if not deleteSelf:
            return
        raiseOnError = raiseOnError if not raiseOnError is None else self._raiseOnError
        rest = list( os.scandir(self._path[:-1]) )
        txt = str(rest)
        txt = txt if len(txt) < 50 else (txt[:47] + '...')
        if len(rest) > 0:
            _log.verify( not raiseOnError, "Cannot delete my own directory %s: directory not empty: found %ld object(s): %s", self._path,len(rest), txt)
            return
        os.rmdir(self._path[:-1])   ## does not work ????
        self._path = None
            
    def eraseEverything( self, keepDirectory = True ):
        """
        Deletes the entire sub directory will all contents
        WARNING: deletes ALL files, not just those with the present extension.
        Will keep the subdir itself by default.
        If not, it will invalidate 'self._path'
        
        If self is None, do nothing. That means you can call this function several times.
        """
        if self._path is None:
            return
        shutil.rmtree(self._path[:-1], ignore_errors=True)
        if not keepDirectory and os.path.exists(self._path[:-1]):
            os.rmdir(self._path[:-1])
            self._path = None
        elif keepDirectory and not os.path.exists(self._path[:-1]):
            os.makedirs(self._path[:-1])

    # -- file ops --
    
    def exists(self, key ):
        """ Checks whether 'key' exists. Works with iterables """
        # vector version
        if not isinstance(key,str):
            _log.verify( isinstance(key, Collection), "'key' must be a string or an interable object. Found type %s", type(key))
            return [ self.exists(k) for k in key ]
        # empty directory
        if self._path is None:
            return False
        # single key
        fullFileName = self.fullKeyName(key)
        if not os.path.exists(fullFileName):
            return False
        if not os.path.isfile(fullFileName):
            raise _log.Exceptn("Structural error: key %s: exists, but is not a file (full path %s)",rel=key,abs=fullFileName)
        return True

    def getCreationTime( self, key ):
        """ returns the creation time of 'key', or None if file was not found """
        # vector version
        if not isinstance(key,str):
            _log.verify( isinstance(key, Collection), "'key' must be a string or an interable object. Found type %s", type(key))
            return [ self.getCreationTime(k) for k in key ]
        # empty directory
        if self._path is None:
            return None
        # single key
        fullFileName = self.fullKeyName(key)
        if not os.path.exists(fullFileName):
            return None
        return datetime.datetime.fromtimestamp(os.path.getctime(fullFileName))
    
    # -- dict-like interface --

    def __call__(self, keyOrSub, default = RETURN_SUB_DIRECTORY ):
        """
        Return either the value of a sub-key (file), or return a new sub directory.
        If only one argument is used, then this function returns a new sub directory.
        If two arguments are used, then this function returns read( keyOrSub, default ).

        Member access:                                
            sd  = SubDir("!/test")
            x   = sd('x', None)                      reads 'x' with default value None           
            x   = sd('sd/x', default=1)              reads 'x' from sub directory 'sd' with default value 1
        Create sub directory:
            sd2 = sd("subdir")                       creates and returns handle to subdirectory 'subdir'
            sd2 = sd("subdir1/subdir2")              creates and returns handle to subdirectory 'subdir1/subdir2'
            
        Parameters:
        -----------
            keyOrSub:
                identify the object requested. Should be a string, or a list.
            default:
                If specified, this function reads 'keyOrSub' with read( keyOrSub, default )
                If not specified, then this function calls subDir( keyOrSub ).

        Returns
        -------
            Either the value in the file, a new sub directory, or lists thereof.
            Returns None if an element was not found.
        """
        if default == SubDir.RETURN_SUB_DIRECTORY:
            if not isinstance(keyOrSub, str):
                _log.verify( isinstance(keyOrSub, Collection), "'keyOrSub' must be a string or an iterable object. Found type '%s;", type(keyOrSub))
                return [ SubDir(k,parent=self) for k in keyOrSub ]
            return SubDir(keyOrSub,parent=self)                    
        return self.read( key=keyOrSub, default=default, raiseOnError=False )

    def __getitem__( self, key ):
        """
        Reads 'key'.
        If 'key' does not exist, return None or throw an error if self._raiseOnError is True
        """
        return self.read( key=key, default=None, raiseOnError=self._raiseOnError )

    def __setitem__( self, key, value):
        """ Writes 'value' to 'key' """
        self.write(key,value)
        
    def __delitem__(self,key):
        """
        Delete 'key'.
        Errors handled according to self._raiseOnError
        """
        self.delete(key, raiseOnError=self._raiseOnError )

    def __len__(self) -> int:
        """ Return the number of files (keys) in this directory """
        return len(self.keys())
        
    def __iter__(self):
        """ like dict """
        return self.keys().__iter__()
    
    def __contains__(self, key):
        """ implements 'in' operator """
        return self.exists(key)

    # -- object like interface --

    def __getattr__(self, key):
        """ Allow using member notation to get data """
        return self.read( key=key, default=None, raiseOnError=self._raiseOnError )
        
    def __setattr__(self, key, value):
        """
        Allow using member notation to write data
        Note: keys starting with '_' are /not/ written to disk
        """
        if key[0] == '_':
            self.__dict__[key] = value
        else:   
            self.write(key,value)

    def __delattr__(self, key):
        """ Allow using member notation to delete data """
        _log.verify( key[:1] != "_", "Deleting protected or private members disabled. Fix __delattr__ to support this")
        return self.delete( key=key, raiseOnError=self._raiseOnError )
        
    # -- automatic caching --
    
    def cache(self, f, cacheName = None, cacheSubDir = None):
        """ Decorater to create an auto-matically cached version of 'f'.
        The function will compute a uniqueHash() accross all 'vargs' and 'kwargs'
        Using MD5 to identify the call signature.
        
        autoRoot = Root("!/caching")
        
        @autoRoot.cache
        def my_function( x, y ):
            return x*y
      
        Advanced arguments
           cacheName        : specify name for the cache for this function.
                              By default it is the name of the function
           cacheSubDir      : specify a subdirectory for the function directory
                              By default it is the module name
                              
        When calling the resulting decorate functions, you can use
           caching='yes'    : default, caching is on
           caching='no'     : no caching
           caching='clear'  : delete existing cache. Do not update
           caching='update' : update cache.
                
        The function will set properties afther the function call:
           cached           : True or False to indicate whether cached data was used
           cacheArgKey      : The hash key for this particular set of arguments
           cacheFullKey     : Full key path
           
        my_function(1,2)
        print("Result was cached " if my_function.cached else "Result was computed")
        
        *WARNING*
        The automatic internal file structure is cut off at 64 characters to ensure directory
        names do not fall foul of system limitations.
        This means that the directory for a function may not be unique. Note that the hash
        key for the arguments includes the function and module name, therefore that is
        unique within the limitations of the hash key.
        """
        f_subDir = self.subDir(f.__module__[0:64] if cacheSubDir is None else cacheSubDir)
        f_subDir = f_subDir.subDir(f.__name__[0:64] if cacheName is None else cacheName)
    
        @wraps(f)
        def wrapper(*vargs,**kwargs):
            caching = 'yes'
            if 'caching' in kwargs:            
                caching = kwargs['caching'].lower()
                del kwargs['caching']            
            # simply no caching
            if caching == 'no':
                wrapper.cached = False
                wrapper.cacheArgKey = None
                wrapper.cacheFullKey = None
                return f(*vargs,**kwargs)   
            _log.verify( caching in ['yes','clear','update'], "'caching': argument must be 'yes', 'no', 'clear', or 'update'. Found %s", caching )
            # compute key
            key = uniqueHash(f.__module__, f.__name__,vargs,kwargs)
            wrapper.cacheArgKey = key
            wrapper.cacheFullKey = f_subDir.fullKeyName(key)
            wrapper.cached = False
            # clear?
            if caching != 'yes':
                f_subDir.delete(key)
            if caching == 'clear':
                return f(*vargs,**kwargs)            
            # use cache
            if caching == 'yes' and key in f_subDir:
                wrapper.cached = True
                return f_subDir[key]
            value = f(*vargs,**kwargs)
            f_subDir.write(key,value)
            f_subDir[key] = value
            return value
    
        return wrapper

"""
Default root directories
"""
TempRoot = SubDir("!")
UserRoot = SubDir("~")

def uniqueFileName( length ):
    """ Computes a unique hash'd file name from 'id' """
    def unique_filename(*args, **argv ):
        """
        Returns a unique filename of tghe specified len for the provided arguments
        If the first argument is a string, and within 'length', then return that string.
        """
        if len(argv) == 0 and len(args) == 1 and isinstance(args[0], str):
            if len(args[0]) <= length:
                return args[0]
        uid = uniqueHash(args,argv).encode('utf-8')
        if len(uid) <= length:
            return uid.decode()
        m = shake_128()
        m.update(uid)
        f = m.hexdigest(length//2)
        return f.decode()
    unique_filename.length = length
    return unique_filename

def uniqueFileName32( *args, **argv ) -> str:
    """ Compute a unique ID of length 32 for the provided arguments """
    return uniqueFileName(32)(*args,**argv)

def uniqueFileName48( *args, **argv ) -> str:
    """ Compute a unique ID of length 48 for the provided arguments """
    return uniqueFileName(48)(*args,**argv)

def uniqueFileName64( *args, **argv ) -> str:
    """ Compute a unique ID of length 64 for the provided arguments """
    return uniqueFileName(64)(*args,**argv)


            

    
class CacheMode(object):
    """
    CacheMode
    An object which represents a caching strategy:
    
                                                on    off     renew    clear   readonly
        load upon start from disk if exists     x     -       -        -       x
        write updates to disk                   x     -       x        -       -
        delete existing object upon start       -     -       -        x       -
        
    See cdxbasics.subdir for functions to manage files.
    """
    
    ON = "on"
    OFF = "off"
    RENEW = "renew"
    CLEAR = "clear"
    READONLY = "readonly"
    
    MODES = [ ON, OFF, RENEW, CLEAR, READONLY ]
    HELP = "'on' for standard caching; 'off' to turn off; 'renew' to overwrite any existing cache; 'clear' to clear existing caches; 'readonly' to read existing caches but not write new ones"
    
    def __init__(self, mode : str = None ):
        """
        CachingMode
        An object which represents a caching strategy:

                                                    on    off     renew    clear   readonly
            load upon start from disk if exists     x     -       -        -       x
            write updates to disk                   x     -       x        -       -
            delete existing object upon start       -     -       -        x       -
            
        Parameters:
        -----------
            mode : str
                Which mode to use.
        """
        mode      = self.ON if mode is None else mode
        self.mode = mode.mode if isinstance(mode, CacheMode) else str(mode)
        _log.verify( self.mode in self.MODES, "Caching mode must be 'on', 'off', 'renew', 'clear', or 'readonly'. Found %s", self.mode )
        self._read   = self.mode in [self.ON, self.READONLY]
        self._write  = self.mode in [self.ON, self.RENEW]
        self._delete = self.mode == self.CLEAR
        
    @property
    def read(self) -> bool:
        """ Whether to load any existing data when starting """
        return self._read
    
    @property
    def write(self) -> bool:
        """ Whether to write cache data to disk """
        return self._write
    
    @property
    def delete(self) -> bool:
        """ Whether to delete existing data """
        return self._delete

    def __str__(self) -> str:# NOQA
        return self.mode
    def __repr__(self) -> str:# NOQA
        return self.mode
        
    def __eq__(self, other) -> bool:# NOQA
        return self.mode == other
    def __neq__(self, other) -> bool:# NOQA
        return self.mode != other
    
    @property
    def is_off(self) -> bool:
        """ Whether this cache mode is OFF """
        return self.mode == self.OFF

    @property
    def is_on(self) -> bool:
        """ Whether this cache mode is ON """
        return self.mode == self.ON

    @property
    def is_renew(self) -> bool:
        """ Whether this cache mode is RENEW """
        return self.mode == self.RENEW

    @property
    def is_clear(self) -> bool:
        """ Whether this cache mode is CLEAR """
        return self.mode == self.CLEAR

    @property
    def is_readonly(self) -> bool:
        """ Whether this cache mode is READONLY """
        return self.mode == self.READONLY


    
