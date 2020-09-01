from __future__ import print_function
import sys

# ABSOLUTE_PATH = '/Users/quentin/work/turbulence/data_drone/'
ABSOLUTE_PATH = '/Users/quentin/work/turbulence/data_drone2/'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class Progress(object):
    def __init__(self, max_iter, end_print=''):
        self.max_iter = max_iter
        self.end_print = end_print
        self.iter = 0

    def update_pgr(self, iteration=None):
        if iteration is not None:
            self.iter = iteration
        # Progression
        print('\rProgression : {:0.02f}%'.format((self.iter + 1) * 100 / self.max_iter), end='')
        sys.stdout.flush()
        if self.iter + 1 == self.max_iter:
            print(self.end_print)
        self.iter += 1
