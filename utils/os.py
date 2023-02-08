import os
import platform
import time

from utils import GlobalLogger as logger

def join_path(*ps, is_dir=False):
    ret = ""
    for p in ps[:-1]:
        p = str(p)      # make int and other types compatible
        if not p.endswith('/'):
            p += "/"
        ret += p
    ps_tail = str(ps[-1])   # make int and other types compatible
    if is_dir and not ps_tail.endswith('/'):
        return ret + ps_tail + '/'
    else:
        return ret + ps_tail

def try_create_directory(path, wait_sec=5):
    if os.path.exists(path):
        logger.warning('os.try_create_folder',
                       f'folder to be created: "{path}", has already existed. \n' +
                       'It will be deleted and a new one will be created instead after 5 seconds')
        time.sleep(wait_sec)
        rm_all(path)
    # rm_all only remove contents, so the directory still exists and need not to be created
    else:
        os.mkdir(path)

def rm_all(path):
    if path[-1] == '/':
        path = path[:-1]

    system = platform.system()
    if system == 'Linux':
        assert os.system(f"rm -rf {path}/*") == 0, "Fail to run rmAll"
    elif system == 'Windows':
        win_style_path = path.replace('/', '\\')
        # On windows platform, we must delete all contents in the directory and remake a new empty one
        assert os.system(rf"rmdir /s/q {win_style_path}") == 0, "Fail to run rmAll"
        os.mkdir(win_style_path)
    else:
        raise NotImplementedError(f"Not supported system for rmAll: {system}")