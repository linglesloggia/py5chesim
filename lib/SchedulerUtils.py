#!/usr/bin/env python3prbs_allocate
# -*- coding: utf-8 -*-

"""
SchedulerUtils.py contains the necessary tools for the development of schedulers.

In this module, scheduling algorithms are performed based on a layered 
processing model. Several functions were adapted from the project 
by Davyt, E, Muzante, A y Rizzo, M. (2021.). A5iGnator : 'Framework para 
la implementación de algoritmos de asignación de recursos en 5G. Tesis de 
grado'. Universidad de la República (Uruguay). Facultad de Ingeniería.

"""

import math
import random

import numpy as np
from lib.uestats import past_throughputs_update, has_packets


def calculateUEfactor(obj: 'framework_UEfactor', data_matrix: np.ndarray) -> np.ndarray:
    """
    Apply scheduling algorithms based on a layered processing model.

    Calculates and returns a vector corresponding to the metrics of each user.

    :param obj: Object of type C{framework_UEfactor}, which contains the
        scheduling algorithm in the format specified by the A5iGnator project.
    :type obj: framework_UEfactor
    :param data_matrix: Matrix where each row vector corresponds to the
        parameters that describe the behavior of users on the network. The user
        is free to choose the type of parameter they want, for example a row can
        represent CQI (channel quality indicator) and each column of said row the
        functional value for that parameter for each UE. The dimension of the
        array must be consistent with the layer parameters of the `framework_UEfactor`
        object. For example, if the `framework_UEfactor` object has a layer of MxN,
        `data_matrix` must be NxL.
    :type data_matrix: numpy.ndarray
    :return: Matrix where each of its elements is the factor for each UE according
        to the chosen metric.
    :rtype: numpy.ndarray
    """
    
    # Perform prefunctions on the input data
    result_prev = np.array(data_matrix)
    for n in range(len(obj.layers)):
        for h in range(len(obj.layers[n].prefunctions)):
            if obj.layers[n].prefunctions[h] == "exp":
                result_prev[h, :] = [math.exp(x) for x in result_prev[h, :]]
            if obj.layers[n].prefunctions[h] == "log":
                result_prev[h, :] = [math.log10(x) for x in result_prev[h, :]]
            if obj.layers[n].prefunctions[h] == "sin":
                result_prev[h, :] = [math.sin(x) for x in result_prev[h, :]]
            if obj.layers[n].prefunctions[h] == "cos":
                result_prev[h, :] = [math.cos(x) for x in result_prev[h, :]]

        # Perform either a scaling or a power operation on the input data
        if obj.layers[n].type_layer == "S":
            result_prev = np.matmul(obj.layers[n].matrix, result_prev)
        if obj.layers[n].type_layer == "P":
            result_current = result_prev  
            result = []  
            for g in range(len(obj.layers[n].matrix)):
                result_current[g, :] = np.power(result_prev[g, :], obj.layers[n].matrix[g])
            for h in range(result_current.shape[1]):
                result.append(np.prod(result_current[:, h]))
            result_prev = result

    return result_prev



def assignUEfactor(self, factors):
    """
    Associates the vector of factors obtained from L{calculateUEfactor} as an attribute to each UE object
    defined in intraSliceSch.py. This association is necessary to later work with the UE objects, for example 
    in the L{findMaxFactor} function.

    @param factors: Array of factors obtained from L{calculateUEfactor}.
    @type factors: list or numpy.ndarray
    @return: None
    """
    n = 0
    for ue in list(self.ues.keys()):
        self.ues[ue].pfFactor = factors[n]
        n += 1


def findMaxFactor(self, ind_u):
    """
    Returns the UE with the highest metric and its corresponding ind_u value.

    @param ind_u: Represents the index of the last UE assigned, when it was randomly assigned. 
    When using this function, the user must save the returned value and then use it as a parameter 
    when calling this function.
    @type ind_u: int
    @return: The UE object with the highest metric and the updated value of ind_u
    @rtype: tuple(ue, int)
    """
    factorMax = 0
    factorMaxInd = ""
    for ue in list(self.ues.keys()):
        if len(self.ues[ue].bearers[0].buffer.pckts) > 0 and self.ues[ue].pfFactor > factorMax:
            factorMax = self.ues[ue].pfFactor
            factorMaxInd = ue
    if factorMaxInd == "":
        ue = list(self.ues.keys())[ind_u]
        q = 0
        while not has_packets(self.ues[ue]) and q < len(self.ues):
            if ind_u < len(list(self.ues.keys())) - 1:
                ind_u = ind_u + 1
            else:
                ind_u = 0
            ue = list(self.ues.keys())[ind_u]
            q = q + 1
        factorMaxInd = ue

    return self.ues[factorMaxInd], ind_u


def prbs_allocate(self, maxInd, window_size, band):
    """
    Perform the allocation of PRBs to the UE indicated by `maxInd`, obtained from `findMaxFactor()`, and update the throughput window for each UE.

    @param maxInd: The UE index to assign the PRBs.
    @type maxInd: int
    @param window_size: The length of the moving average window.
    @type window_size: int
    @param band: The resources to host.
    @type band: int
    """
    for ue in (self.ues):
        if ue == maxInd.id:
            self.ues[ue].prbs = band
        else:
            self.ues[ue].prbs = 0
            past_throughputs_update(self.ues[ue], window_size)
            self.ues[ue].tbsz = 1

def sort_by_pf_factor(self):
    """
    Sorts self.ues by each element's pfFactor attribute in descending order.
    
    :return: A sorted list of UE elements.
    """
    sorted_list = sorted(self.ues, key=lambda ue: ue.pfFactor, reverse=True)
    return sorted_list


def allocate_distributed_prbs(self, SliceM, nalloc):
    """
    Perform the allocation of PRBs to the UE indicated by `maxInd`, obtained from `findMaxFactor()`, and update the throughput window for each UE.

    @param maxInd: The UE index to assign the PRBs.
    @type maxInd: int
    @param window_size: The length of the moving average window.
    @type window_size: int
    @param band: The resources to host.
    @type band: int
    """
    for ue in (self.ues):
        if ue == maxInd.id:
            self.ues[ue].prbs = band
        else:
            self.ues[ue].prbs = 0
            past_throughputs_update(self.ues[ue], window_size)
            self.ues[ue].tbsz = 1

        
class layer:
    """
    Define the characteristics of a layer for use in scheduling logic.

    A layer is the most basic unit of layered scheduling logic, which contains
    the necessary information on the operations to be carried out on its layer.
    Each layer has the possibility of applying different types of functions in
    order to achieve more complex scheduling algorithms, therefore each layer
    has the possibility of carrying out a prefunctions stage, where said function
    is applied to each input prior to manipulation with the coefficients of
    the layer.

    @ivar type_layer: The type of processing that will be carried out in the layer. Can be 'P' or 'S', where 'P' means there will be a production of the input elements raised to the coefficients of the layer, and 'S' means that within the layer a linear combination of the weighted inputs to the layer coefficients.
    @type type_layer: str
    @ivar matrix: The layer coefficients for either 'P' or 'S' type of layers. If it is of the 'P' type, the coefficients represent the value to which each attribute is raised in the scheduler input. If it is of the 'S' type, the coefficients are the weights of each input.
    @type matrix: list of list of float
    @ivar prefunctions: The prefunctions to be carried out prior to the operation of the layer. Can take the values 'exp', 'log', 'cos', 'sin', and 'id'.
    @type prefunctions: str
    """
    def __init__(self, type_layer, matrix, prefunctions=0):
        """
        Initialize a layer with the given `type_layer`, `matrix`, and `prefunctions`.

        @param type_layer: The type of processing that will be carried out in the layer. Can be 'P' or 'S', where 'P' means there will be a production of the input elements raised to the coefficients of the layer, and 'S' means that within the layer a linear combination of the weighted inputs to the layer coefficients.
        @type type_layer: str
        @param matrix: The layer coefficients for either 'P' or 'S' type of layers. If it is of the 'P' type, the coefficients represent the value to which each attribute is raised in the scheduler input. If it is of the 'S' type, the coefficients are the weights of each input.
        @type matrix: list of list of float
        @param prefunctions: The prefunctions to be carried out prior to the operation of the layer. Can take the values 'exp', 'log', 'cos', 'sin', and 'id'.
        @type prefunctions: str
        """
        self.type_layer = type_layer
        """ 'P' or 'S' """
        self.matrix = matrix
        """ Layer coefficients """
        self.prefunctions = prefunctions
        """ "exp", "log", "cos" , "sin", "id" """


class framework_UEfactor:
    """
    Model for scheduling operations using layers.

    The scheduling algorithms are implemented using C{framework_UEfactor}
    objects, which contain the layers needed to perform the desired scheduling
    algorithm. For more information and examples, refer to the C{a5ignator}
    documentation.
    """

    def __init__(self, layers):
        """
        Constructor.

        @param layers: An array of L{layer} objects. Each layer performs
        processing on the input data.
        @type layers: list of L{layer}
        """
        self.layers = layers
        """Saves the layers as an attribute."""
        self.__layer_types = ["P", "S"]
        """The types of layers, 'P' or 'S'."""
        self.__available_prefunctions = ["exp", "log", "cos", "sin", "id"]
        """The available prefunctions, which are 'exp', 'log', 'cos', 'sin', or 'id'."""

    def validate(self):
        """
        Checks if the configuration is valid.

        @return: True if the configuration is valid, False otherwise.
        @rtype: bool
        """
        for layer in self.layers:
            if layer.type_layer not in self.__layer_types:
                print("Error: There is a layer that is neither 'S' nor 'P'")
                return False
            if layer.prefunctions not in self.__available_prefunctions:
                print("Error: There is a layer with an invalid prefunction")
                return False
        return True




# for slices

def calculateSlicefactor(obj, data_matrix):  # calculateUEfactor
    """
    It is used to apply scheduling algorithms.

    Calculates and returns a vector corresponding to the metrics of each user.

    @param obj: object of type C{framework_slicefactor}, which contains the
    scheduling algorithm in the format specified by the a5gnator project.
    @param data_matrix: matrix where each row vector corresponds to the
    parameters that describe the behavior of users on the network. The user
    is free to choose the type of parameter they want, for example a row can
    represent CQI (channel quality indicator) and each column of said row the
    functional value for that parameter for each UE. The dimension of the
    array must be consistent with the layer parameters of the
    L{framework_slicefactor} object, e.g. if the L{framework_slicefactor} object has
    a layer of MxN, C{data_matrix} must be NxL.
    @return: matrix where each of its elements is the factor for each ue
    according to the chosen metric.
    """

    result_prev = np.array(data_matrix)

    for n in range(len(obj.layers)):
        for h in range(len(obj.layers[n].prefunctions)):
            if obj.layers[n].prefunctions[h] == "exp":
                result_prev[h, :] = [math.exp(x) for x in result_prev[h, :]]

            if obj.layers[n].prefunctions[h] == "log":
                result_prev[h, :] = [math.log10(x) for x in result_prev[h, :]]

            if obj.layers[n].prefunctions[h] == "sin":
                result_prev[h, :] = [math.sin(x) for x in result_prev[h, :]]

            if obj.layers[n].prefunctions[h] == "cos":
                result_prev[h, :] = [math.cos(x) for x in result_prev[h, :]]

        if obj.layers[n].type_layer == "S":
            result_prev = np.matmul(obj.layers[n].matrix, result_prev)

        if obj.layers[n].type_layer == "P":
            result_current = result_prev 
            result = []  
            for g in range(len(obj.layers[n].matrix)):
                
                result_current[g, :] = np.power(
                    result_prev[g, :], obj.layers[n].matrix[g]
                )
            for h in range(result_current.shape[1]):
                result.append(np.prod(result_current[:, h]))


            result_prev = result

    return result_prev

def assignSlicefactor(self, factors):
    """
    Relates the result of setUEfactor to the array of UEs.

    associates the vector of factors (obtained for example from
    calculateUEfactor), as an attribute to each ue. this association is
    necessary to later work with the ue objects, for example in the
    L{findMaxFactor} function.

    @param ues: array of ues defined in intraSliceSch.py.
    @param factors: array of factors obtained from L{calculateUEfactor}.
    @return: assigns each ue the value of its factor.
    """
    n = 0
    for slice in list(self.slices.keys()):
        self.slices[slice].metric = factors[n]
        n += 1
    return


def findMaxMetSlice(self):
    """This method finds and returns the Slice with the highest metric"""
    metric = 0
    for slice in list(self.slices.keys()):
        if self.slices[slice].metric > metric:
            metric = self.slices[slice].metric
            maxSliceM = slice
        if self.slices[slice].metric == metric:
            slicesMlist = [maxSliceM, slice]
            maxSliceM = random.choice(slicesMlist)
    return maxSliceM


def slice_prbs_allocate(self, maxSliceM):
    """
    This function performs the hosting of prbs to the ues.

    assigns the prbs corresponding to the ue indicated by C{maxInd} (obtained
    from findMaxFactor). It also updates the throghput window passed for each ue.

    @param band: resources to host.
    @param maxInd: ue index to assign the prbs.
    @param ues: array of ues defined in intraSliceSch.py.
    @type ues: ue array.
    @param window_size: moving average window length.
    @type window_size: integer.


    """
    """This method allocates cell's PRBs to the indicated slice"""
    for sl in list(self.slices.keys()):
        if sl == maxSliceM:
            self.slices[sl].updateConfig(int(self.PRBs / self.slices[sl].numRefFactor))
        else:
            self.slices[sl].updateConfig(0)

def slice_allocate_distributed_prbs(self, SliceM, nalloc):
    """
    This function performs the hosting of prbs to the ues.

    assigns the prbs corresponding to the ue indicated by C{maxInd} (obtained
    from findMaxFactor). It also updates the throghput window passed for each ue.

    @param band: resources to host.
    @param maxInd: ue index to assign the prbs.
    @param ues: array of ues defined in intraSliceSch.py.
    @type ues: ue array.
    @param window_size: moving average window length.
    @type window_size: integer.


    """
    """This method allocates cell's PRBs to the indicated slice"""
    for sl in list(self.slices.keys()):
        if sl == SliceM:
            self.slices[sl].updateConfig(int(nalloc / self.slices[sl].numRefFactor))
        else: # here depends if the user wants all in one or other aproach
            self.slices[sl].updateConfig(0)

class slice_framework:
    """
    Operations layer model for scheduling.

    the scheduling algorithms are implemented through C{framework_UEfactor}
    objects, these objects contain the necessary layers to carry out the
    scheduling algorithm of interest.
    For more information and examples consult the C{a5ignator} documentation.
    """

    def __init__(self, layers):
        """
        Constructor.

        @param layers: array of L{layer} objects, each layer will do processing
        on the data.
        """
        self.layers = layers
        """Save the layers as an attribute."""
        self.__layer_types = ["P", "S"]
        self.__available_prefunctions = ["exp", "log", "cos", "sin", "id"]

    def validate(self):
        """
        Function that returns if the configuration is valid or not.

        @return: True if the configuration is valid, False if not.
        """
        for layer in self.layers:
            if layer.type_layer not in self.__layer_types:
                print("Error: There is a layer that is neither 'S' nor 'P'")
                return False
            if layer.prefunctions not in self.__available_prefunctions:
                print(
                    "Error: There is a layer whose prefunction is not in the available prefunctions (exp, log, cos, sin, id)."
                )
                return False
        return True
# other useful functions

def getTbs(Ninfo: int, r: float) -> int:
    """
    Calculates the transport block size (TBS) for a given amount of information to be transmitted and target code rate.

    :param Ninfo: An integer representing the amount of information to be transmitted.
    :param r: A float representing the target code rate.
    :return: An integer representing the calculated TBS value.

    If Ninfo is less than 24, return 0. If Ninfo is between 24 and 3824, find the largest TBS value in the TBS table
    that is less than Ninfo and return it. If Ninfo is greater than 3824, calculate TBS value based on the given formula.
    If r is less than or equal to 1/4, calculate TBS value based on the given formula. If r is greater than 1/4,
    calculate TBS value based on the given formula. Return the calculated TBS value.
    """
        
    # If the amount of information to be transmitted is less than 24, return 0
    if Ninfo < 24:
        return 0

    # If the amount of information to be transmitted is between 24 and 3824, find the largest TBS (Transport Block Size) value in the TBS table
    # that is less than Ninfo and return it
    if Ninfo <= 3824:
        for i in range(len(self.tbsTable)):
            if Ninfo < self.tbsTable[i]:
                return self.tbsTable[i-1]

    # If the amount of information to be transmitted is greater than 3824, calculate TBS value based on the given formula
    else:
        # Calculate n based on the formula
        n = np.floor(np.log2(Ninfo-24))-5
        
        # Calculate Ninfo_ based on the formula
        Ninfo_ = max(3840, pow(2,n) * np.round((Ninfo - 24)/(pow(2,n))) )
        
        # If r is less than or equal to 1/4, calculate TBS value based on the given formula
        if r <= 1/4:
            C = np.ceil((Ninfo_/+24)/3816)
            TBS = 8*C*np.ceil((Ninfo_+24)/(8*C)) - 24
            
        # If r is greater than 1/4, calculate TBS value based on the given formula
        else:
            # If Ninfo_ is greater than 8424, calculate C based on the given formula
            if Ninfo_ > 8424:
                C = np.ceil((Ninfo_+24)/8424)
                TBS = 8*C*np.ceil((Ninfo_+24)/(8*C)) - 24
            # If Ninfo_ is less than or equal to 8424, calculate TBS value based on the given formula
            else:
                TBS = 8*np.ceil((Ninfo_+24)/8) - 24

    # Return the calculated TBS value
    return TBS
    


def loadModTable():
    """MCS table 2 (5.1.3.1-2) from 3GPP TS 38.214"""
    modTable = []
    modTable.append(
        {
            "spctEff": 0.2344,
            "bitsPerSymb": 2,
            "codeRate": 0.1171875,
            "mcsi": 0,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.377,
            "bitsPerSymb": 2,
            "codeRate": 0.1884765625,
            "mcsi": 1,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.6016,
            "bitsPerSymb": 2,
            "codeRate": 0.30078125,
            "mcsi": 2,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.877,
            "bitsPerSymb": 2,
            "codeRate": 0.4384765625,
            "mcsi": 3,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.1758,
            "bitsPerSymb": 2,
            "codeRate": 0.587890625,
            "mcsi": 4,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.4766,
            "bitsPerSymb": 4,
            "codeRate": 0.369140625,
            "mcsi": 5,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.6953,
            "bitsPerSymb": 4,
            "codeRate": 0.423828125,
            "mcsi": 6,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.9141,
            "bitsPerSymb": 4,
            "codeRate": 0.478515625,
            "mcsi": 7,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.1602,
            "bitsPerSymb": 4,
            "codeRate": 0.5400390625,
            "mcsi": 8,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.4063,
            "bitsPerSymb": 4,
            "codeRate": 0.6015625,
            "mcsi": 9,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.5703,
            "bitsPerSymb": 4,
            "codeRate": 0.642578125,
            "mcsi": 10,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.7305,
            "bitsPerSymb": 6,
            "codeRate": 0.455078125,
            "mcsi": 11,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.0293,
            "bitsPerSymb": 6,
            "codeRate": 0.5048828125,
            "mcsi": 12,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.3223,
            "bitsPerSymb": 6,
            "codeRate": 0.5537109375,
            "mcsi": 13,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.6094,
            "bitsPerSymb": 6,
            "codeRate": 0.6015625,
            "mcsi": 14,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.9023,
            "bitsPerSymb": 6,
            "codeRate": 0.650390625,
            "mcsi": 15,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.2129,
            "bitsPerSymb": 6,
            "codeRate": 0.7021484375,
            "mcsi": 16,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.5234,
            "bitsPerSymb": 6,
            "codeRate": 0.75390625,
            "mcsi": 17,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.8164,
            "bitsPerSymb": 6,
            "codeRate": 0.802734375,
            "mcsi": 18,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.1152,
            "bitsPerSymb": 6,
            "codeRate": 0.8525390625,
            "mcsi": 19,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.332,
            "bitsPerSymb": 8,
            "codeRate": 0.66650390625,
            "mcsi": 20,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.5547,
            "bitsPerSymb": 8,
            "codeRate": 0.6943359375,
            "mcsi": 21,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.8906,
            "bitsPerSymb": 8,
            "codeRate": 0.736328125,
            "mcsi": 22,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.2266,
            "bitsPerSymb": 8,
            "codeRate": 0.7783203125,
            "mcsi": 23,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.5703,
            "bitsPerSymb": 8,
            "codeRate": 0.8212890625,
            "mcsi": 24,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.9141,
            "bitsPerSymb": 8,
            "codeRate": 0.8642578125,
            "mcsi": 25,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 7.1602,
            "bitsPerSymb": 8,
            "codeRate": 0.89501953125,
            "mcsi": 26,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 7.4063,
            "bitsPerSymb": 8,
            "codeRate": 0.92578125,
            "mcsi": 27,
            "mod": "256QAM",
        }
    )
    return modTable

def loadtbsTable():
    tbsTable = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 
                    144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 
                    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 
                    704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 
                    1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 
                    1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 
                    2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824]
    return tbsTable

def loadSINR_MCStable(tdd):
    """MCS-SINR allocation table"""
    sinrModTable = []

    if tdd:
        ############### TDD #########################
        sinrModTable.append(1.2)  # MCS 0
        sinrModTable.append(3.9)  # MCS 1
        sinrModTable.append(4.9)  # MCS 2
        sinrModTable.append(7.1)  # MCS 3
        sinrModTable.append(7.8)  # MCS 4
        sinrModTable.append(9.05)  # MCS 5
        sinrModTable.append(10.0)  # MCS 6
        sinrModTable.append(11.1)  # MCS 7
        sinrModTable.append(12.0)  # MCS 8
        sinrModTable.append(13.2)  # MCS 9
        sinrModTable.append(14.0)  # MCS 10
        sinrModTable.append(15.2)  # MCS 11
        sinrModTable.append(16.1)  # MCS 12
        sinrModTable.append(17.2)  # MCS 13
        sinrModTable.append(18.0)  # MCS 14
        sinrModTable.append(19.2)  # MCS 15
        sinrModTable.append(20.0)  # MCS 16
        sinrModTable.append(21.8)  # MCS 17
        sinrModTable.append(22.0)  # MCS 18
        sinrModTable.append(22.5)  # MCS 19
        sinrModTable.append(22.9)  # MCS 20
        sinrModTable.append(24.2)  # MCS 21
        sinrModTable.append(25.0)  # MCS 22
        sinrModTable.append(27.2)  # MCS 23
        sinrModTable.append(28.0)  # MCS 24
        sinrModTable.append(29.2)  # MCS 25
        sinrModTable.append(30.0)  # MCS 26
        sinrModTable.append(100.00)  # MCS 27

    else:
        ############## FDD #########################
        sinrModTable.append(0.0)  # MCS 0
        sinrModTable.append(3.0)  # MCS 1
        sinrModTable.append(5.0)  # MCS 2
        sinrModTable.append(7.0)  # MCS 3
        sinrModTable.append(8.1)  # MCS 4
        sinrModTable.append(9.3)  # MCS 5
        sinrModTable.append(10.5)  # MCS 6
        sinrModTable.append(11.9)  # MCS 7
        sinrModTable.append(12.7)  # MCS 8
        sinrModTable.append(13.4)  # MCS 9
        sinrModTable.append(14.0)  # MCS 10
        sinrModTable.append(15.8)  # MCS 11
        sinrModTable.append(16.8)  # MCS 12
        sinrModTable.append(17.8)  # MCS 13
        sinrModTable.append(18.4)  # MCS 14
        sinrModTable.append(20.1)  # MCS 15
        sinrModTable.append(21.1)  # MCS 16
        sinrModTable.append(22.7)  # MCS 17
        sinrModTable.append(23.6)  # MCS 18
        sinrModTable.append(24.2)  # MCS 19
        sinrModTable.append(24.5)  # MCS 20
        sinrModTable.append(25.6)  # MCS 21
        sinrModTable.append(26.3)  # MCS 22
        sinrModTable.append(28.3)  # MCS 23
        sinrModTable.append(29.3)  # MCS 24
        sinrModTable.append(31.7)  # MCS 25
        sinrModTable.append(35.0)  # MCS 26
        sinrModTable.append(100.00)  # MCS 27

    return sinrModTable