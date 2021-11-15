import numpy as np

def get_times(path):
    '''
    Returns fm intervals which are stored in a file
    '''
    times = np.array([], dtype=np.float32)
    with open(path) as f:
        for idx, line in enumerate(f):
            t = np.fromstring(" ".join(line.split()), sep = ' ', dtype=np.float32)
            if len(t) == 1:
                times = np.hstack((times, t))
    return times
