import shelve
import os


def store_model(filepath, key, model):
    try:
        directory = os.path.dirname(filepath)
        os.makedirs(directory)
    except FileExistsError:
        pass

    f = shelve.open(filepath)
    f[key] = model
    f.close()


def retrieve_model(filepath, key):
    f = shelve.open(filepath)
    model = f[key]
    f.close()

    return model
