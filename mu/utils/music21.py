try:
    import music21 as m21
    music21 = True
except ImportError:
    music21 = False


def decorator(func):
    def wrapper(*args, **kwargs):
        if music21 is not False:
            return func(*args, *kwargs)
        else:
            raise ImportError("Can't find package music21")
    return wrapper
