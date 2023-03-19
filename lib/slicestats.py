import sys

#sys.path.append("../")
#from InterSliceSch import InterSliceScheduler


def get_slice_throughput(self, slice):
    """
    This method calculates the total throughput of a given slice. It loops through all the user equipment
    (UE) in the slice and calculates the throughput by calling the setMod method of the downlink scheduler
    for the UE.

    :param slice: The name of the slice.
    :return: The total throughput of the slice.
    """
    throughput = 0
    for ue in list(self.slices[slice].schedulerDL.ues.keys()):
        throughput += self.slices[slice].schedulerDL.setMod(ue, self.PRBs)[0]
    return throughput


def get_slice_past_throughput(self, slice):
    """
    This method calculates the past throughput of a given slice. It uses the received bytes of the first and
    last packets to calculate the total bytes received and then calculates the throughput.

    :param slice: The name of the slice.
    :return: The past throughput of the slice.
    """
    rcvdBt_end = self.slices[slice].rcvdBytes[len(self.slices[slice].rcvdBytes) - 1]
    rcvdBt_in = self.slices[slice].rcvdBytes[0]
    if rcvdBt_end - rcvdBt_in == 0:
        den = 1
    else:
        den = rcvdBt_end - rcvdBt_in
    return den


def get_slice_packets(self, slice):
    """
    This method returns the total number of packets in a given slice.

    :param slice: The name of the slice.
    :return: The total number of packets in the slice.
    """
    return self.slices[slice].schedulerDL.updSumPcks()

def slice_has_packets(slice):
    """
    This method checks if a given slice has any packets or not by calling the get_slice_packets method.

    :param slice: The name of the slice.
    :return: True if the slice has packets, False otherwise.
    """
    if get_slice_packets(slice) > 0:
        return True
    else:
        return False


def set_NSRS(slice):
    """
    This method calculates the Network Slice Resource Satisfaction (NSRS) for a given slice. It calculates
    the ratio of the number of user equipments (UEs) in the slice that have achieved the required throughput
    to the total number of UEs in the slice.

    :param slice: The name of the slice.
    :return: The NSRS of the slice.
    """
    users_satisfied = 0
    n_users = len(self.slices[slice].schedulerDL.ues.keys())
    for ue in list(self.slices[slice].schedulerDL.ues.keys()):
        if (
            self.slices[slice].schedulerDL.ues[ue].throughput
            > self.slices[slice].reqThroughput
        ):
            users_satisfied += 1

    if n_users == 0:
        return 0
    else:
        return users_satisfied / n_users

def set_RBUR(self, slice: int) -> float:
    """
    Sets the RBUR (Resource Block Utilization Ratio) for the given slice by modifying
    the schedulerDL's pks_s and tbSize attributes.
    
    :param slice: An integer representing the index of the slice to modify.
    :return: A float representing the RBUR value for the slice.
    """
    pks_s = self.slices[slice].schedulerDL.pks_s
    tbSize = self.slices[slice].schedulerDL.tbSize

    self.slices[slice].schedulerDL.pks_s = 0
    self.slices[slice].schedulerDL.tbSize = 0

    if (pks_s > tbSize and tbSize != 0) or pks_s == 0:
        # If pks_s is greater than tbSize (and tbSize is non-zero) or if pks_s is zero,
        # set pks_s and tbSize to zero and return 1 (i.e., an RBUR value of 100%).
        return 1
    elif pks_s != 0 and tbSize == 0:
        # If pks_s is non-zero and tbSize is zero, set pks_s and tbSize to zero
        # and return 0 (i.e., an RBUR value of 0%).
        return 0
    else:
        # Otherwise, calculate the RBUR value as pks_s divided by tbSize.
        return pks_s / tbSize