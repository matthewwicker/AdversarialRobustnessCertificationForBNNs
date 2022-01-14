import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import sklearn.model_selection as modsel

######################################################
## PyTorch Dataset for presplit regression datasets ##
######################################################

from data_handlers.boston_presplit import BostonPresplit
from data_handlers.concrete_presplit import ConcretePresplit
from data_handlers.energy_presplit import EnergyPresplit
from data_handlers.kin8nm_presplit import Kin8nmPresplit
from data_handlers.naval_presplit import NavalPresplit
from data_handlers.powerplant_presplit import PowerplantPresplit
from data_handlers.wine_presplit import WinePresplit
from data_handlers.yacht_presplit import YachtPresplit


DEFAULT_DATA_FOLDER = "./data"


################################################
## Construct class for dealing with data sets ##
################################################

class Dataset():
    def __init__(self, data_set, data_folder=DEFAULT_DATA_FOLDER):
        super(type(self), self).__init__()

        if data_set in ["boston" + str(i) for i in range(20)]:
            self.train_set = BostonPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = BostonPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 13
            self.num_classes = None

        elif data_set in ["concrete" + str(i) for i in range(20)]:
            self.train_set = ConcretePresplit(root = data_folder,
                                              data_set = data_set,
                                              train = True)
            self.test_set = ConcretePresplit(root = data_folder,
                                             data_set = data_set,
                                             train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["energy" + str(i) for i in range(20)]:
            self.train_set = EnergyPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = EnergyPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["kin8nm" + str(i) for i in range(20)]:
            self.train_set = Kin8nmPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = True)
            self.test_set = Kin8nmPresplit(root = data_folder,
                                            data_set = data_set,
                                            train = False)

            self.task = "regression"
            self.num_features = 8
            self.num_classes = None

        elif data_set in ["naval" + str(i) for i in range(20)]:
            self.train_set = NavalPresplit(root = data_folder,
                                           data_set = data_set,
                                           train = True)
            self.test_set = NavalPresplit(root = data_folder,
                                          data_set = data_set,
                                          train = False)

            self.task = "regression"
            self.num_features = 16
            self.num_classes = None

        elif data_set in ["powerplant" + str(i) for i in range(20)]:
            self.train_set = PowerplantPresplit(root = data_folder,
                                                data_set = data_set,
                                                train = True)
            self.test_set = PowerplantPresplit(root = data_folder,
                                               data_set = data_set,
                                               train = False)

            self.task = "regression"
            self.num_features = 4
            self.num_classes = None

        elif data_set in ["wine" + str(i) for i in range(20)]:
            self.train_set = WinePresplit(root = data_folder,
                                          data_set = data_set,
                                          train = True)
            self.test_set = WinePresplit(root = data_folder,
                                         data_set = data_set,
                                         train = False)

            self.task = "regression"
            self.num_features = 11
            self.num_classes = None

        elif data_set in ["yacht" + str(i) for i in range(20)]:
            self.train_set = YachtPresplit(root = data_folder,
                                           data_set = data_set,
                                           train = True)
            self.test_set = YachtPresplit(root = data_folder,
                                          data_set = data_set,
                                          train = False)

            self.task = "regression"
            self.num_features = 6
            self.num_classes = None


        else:
            RuntimeError("Unknown data set")
