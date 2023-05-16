import pickle


def save_obj(obj, name):
    """
    A helper function to save pickled objects.
    :params
    obj: The object that needs to be saved.
    name: The file name and sometimes path+filename to save this object as
    :returns
    void
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
