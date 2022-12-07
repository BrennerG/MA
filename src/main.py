import sys
import torch
import experiments.experiment_util as EUT
import logging

'''
# ***************************** MAIN FILE ******************************** #

It's the main file!
Currently this is for handling the Experiment Class, which is the main unit of this repo.
An Experiment is run from start to finish using the given parameter settings.
(Parameter search, generation and reading from .yaml will be supported later)

'''


if __name__ == "__main__":

    # parse arguments
    args = [a.replace('--','').split('=') for a in sys.argv][1:]

    logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

    # run the experiment
    exp = EUT.experiment_factory('qagnn', args)
    exp()
    print('done')