from __future__ import unicode_literals, print_function, division


class EarlyStopConditionByCount():
    def __init__(self, stop_steps, verbose=False):
        self.stop_steps = stop_steps
        self.step_counter = 0
        self.verbose = verbose

    def __call__(self):
        stop = self.step_counter >= self.stop_steps
        if stop and self.verbose:
            print("Early stop by count!")
        return stop

    def incr(self):
        self.step_counter += 1

    def reset(self):
        self.step_counter = 0
