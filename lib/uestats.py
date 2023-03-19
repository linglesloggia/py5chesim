#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
uestats module, contains functions that return statistics of the ues

This module contains functions to obtain statistics for each UE. These 
functions can be used by Scheds_intra.py in conjunction with SchedulerUtils.py 
to obtain complex resource allocation algorithms.
"""

import sys

sys.path.append("../")

def get_queue_size(ue):
    """
    Determines the number of packets queued for a UE.

    @param ue: object of type ue defined in UE.py. An element of the ues array
    defined in intraSliceSch.py ​​is commonly used.
    @returns: length of the packet queue to send.
    @rtype: ue object.
    """
    return len(ue.bearers[0].buffer.pckts)


def has_packets(ue):
    """
    Function that determines whether or not there are packets in the queue.

    @param ue: object of type ue defined in UE.py.
    @return: True if the eu has packets in queue, false otherwise.
    @rtype: bool.
    """
    return True if (get_queue_size(ue) > 0) else False


def past_throughputs_update(ue, window_size):
    """
    Function used to set the moving average.

    assigns or updates the attribute corresponding to the past throughput
    of an ue. Required during resource allocation, e.g. for the proportional
    fair algorithm.

    @param ue: object of type ue defined in UE.py. An element of the ues array
    defined in intraSliceSch.py ​​is commonly used.
    @param window_size: moving average window length.
    @type window_size: integer.
    @return: the function assigns the last throughput value to a queue
    and removes the first one based on the windows_size parameter.
    """

    if len(ue.pastTbsz) > window_size:
        ue.pastTbsz.popleft()
    ue.pastTbsz.append(ue.tbsz)

    pass


def average_past_throughputs(self, ue):
    """
    Returns the average of past throughputs.

    returns the average throughput of a ue. The amount of throughputs saved
    is an attribute of the ue, it can be modified in the function L{past_throughputs_update}
    The average will be determined by the number of elements within
    the variable C{past_throughput}, which goes from 1 to C{windows_size}.

    @param ue: ue type object.
    @type ue: ue.
    @return: the sum of the previous throughputs over the amount of
    elements added.
    @rtype: float.
    """

    accumulated = 0
    for thgt in self.ues[ue].pastTbsz:
        accumulated = accumulated + thgt
    return accumulated / len(self.ues[ue].pastTbsz)


def get_snir(ue):
    """
    Function used to determine the signal-to-noise and interference ratio
    of the C{ue} object.

    @param ue: ue type object.
    @return: snir of the ue user.
    """

    return math.pow(10, ue.radioLinks.linkQuality / 10)


def get_throughput(self, u):
    """
    Returns the throughput value for a UE. In this implementation,
    the transport block size (tbs) is taken as a measure of throughput.

    @param u: ue type object.
    @type u: ue.
    @param nprb: maximum amount of RB for each UE in a TTI.
    @type nprb: integer.
    @return: the amount of current throughput.
    @rtype: integer.
    """
    return self.setMod(u, self.nrbUEmax)[0]


def get_modulation_scheme(self, u, nprb):
    """
    Returns the modulation scheme value for a UE.

    @param u: ue type object.
    @type u: ue.
    @param nprb: maximum amount of RB for each UE in a TTI.
    @type nprb: integer.
    @return: modulation scheme of the ue.
    @rtype: string.
    """
    return self.setMod(u, nprb)[1]


def get_mcsi(self, u, nprb):
    """
    Returns the mcsi value for a UE.

    @param u: ue type object.
    @type u: ue.
    @param nprb: maximum amount of RB for each UE in a TTI.
    @type nprb: integer.
    @return: the mcsi of the ue.
    @rtype: integer.
    """
    return self.setMod(u, nprb)[3]


def get_bits_per_sym(self, u, nprb):
    """
    Returns the bits per symbol of an UE.

    @param u: ue type object.
    @type u: ue.
    @param nprb: maximum amount of RB for each UE in a TTI.
    @type nprb: integer.
    @return: the bits per symbol of an ue.
    @rtype: integer.
    """
    return self.setMod(u, nprb)[2]


def get_loss_probability(self, user):
    """
    Function that calculates the ue probability of packet loss at a certain time.

    """
    return user.packetFlows[0].meassuredKPI["PacketLossRate"]


def uesWithPackets(self):
    uesWithPackets = 0
    for ue in list(self.ues.keys()):
        if len(self.ues[ue].bearers[0].buffer.pckts) > 0:
            uesWithPackets += 1

    return uesWithPackets

def get_delay(ue):
    """
    Functions that returns the maximum delay perceived by an user.
    """
    return ue.delay