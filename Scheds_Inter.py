"""This module contains different implemented inter slice schedulers.
New schedulers should be implemented here following the current structure."""
import math
import pickle
import random

from InterSliceSch import InterSliceScheduler
from Slice import *

sys.path.insert(1, "lib")
from lib.slicestats import get_slice_throughput,get_slice_past_throughput

# from AISchedulersUtils import *


class RRplus_Scheduler(InterSliceScheduler):
    """
    Implements Round Robin Plus inter slice scheduling algorithm.

    :param ba: bandwidth allocation
    :type ba: dict
    :param fr: frequency range
    :type fr: int
    :param dm: domain
    :type dm: str
    :param tdd: time division duplex
    :type tdd: str
    :param gr: granularity
    :type gr: int
    """

    def __init__(self, ba: dict, fr: int, dm: str, tdd: str, gr: int):
        """
        Initialize a new RRplus_Scheduler.

        :param ba: bandwidth allocation
        :type ba: dict
        :param fr: frequency range
        :type fr: int
        :param dm: domain
        :type dm: str
        :param tdd: time division duplex
        :type tdd: str
        :param gr: granularity
        :type gr: int
        """
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)

    def resAlloc(self, env):
        """
        Implement Round Robin Plus PRB allocation between the different configured slices.
        This PEM method overwrites the resAlloc method from InterSliceScheduler class.

        Round Robin Plus scheduler allocates the same amount of resources to each slice with packets in buffer.

        :param env: simpy environment
        :type env: simpy.Environment
        """
        while True:
            self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")
            if len(list(self.slices.keys())) > 0:
                if len(list(self.slices.keys())) > 1:
                    slicesWithPacks = 0
                    for slice in list(self.slices.keys()):
                        sliceHasPackets = (
                            self.slices[slice].schedulerDL.updSumPcks() > 0
                            or self.slices[slice].schedulerUL.updSumPcks() > 0
                        )
                        if sliceHasPackets:
                            slicesWithPacks = slicesWithPacks + 1

                    if slicesWithPacks == 0:
                        for slice in list(self.slices.keys()):
                            self.slices[slice].updateConfig(
                                0
                                )
                            
                            self.printSliceConfig(slice)
                    else:
                        for slice in list(self.slices.keys()):
                            sliceHasPackets = (
                                self.slices[slice].schedulerDL.updSumPcks() > 0
                                or self.slices[slice].schedulerUL.updSumPcks() > 0
                            )
                            if not sliceHasPackets:
                                self.slices[slice].updateConfig(0)
                            else:
                                self.slices[slice].updateConfig(
                                    int(
                                        (self.PRBs / slicesWithPacks)
                                        / self.slices[slice].numRefFactor
                                    )
                                )
                            self.printSliceConfig(slice)
                else:
                    slice = self.slices[list(self.slices.keys())[0]]
                    prbs = 0
                    for b in self.bw:
                        prbs = prbs + self.nRBtable[self.FR][slice.scs][str(b) + "MHz"]
                    slice.updateConfig(prbs)

            self.dbFile.write("<hr>")
            yield env.timeout(self.granularity)



class PF_Scheduler(InterSliceScheduler):
    """This class implements Proportional Fair inter slice scheduling algorithm."""

    def __init__(self, ba, fr, dm, tdd, gr, sch):
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)
        self.sch = sch
        """String formatted as PFXY, with X=numerator exponent for metric formula, and Y=denominator exponent. """
        self.rcvdBytesLen = 10
        """rcvdBytes list length in Slice instance. No more that rcvdBytesLen values are stored."""

        exp_num = float(self.sch[2])
        exp_den = float(self.sch[3])

        layer1 = layer("P", np.array([exp_num, -exp_den]), "id")

        self.scheduler = slice_framework([layer1])

        """object of type framework_UEfactor that contains the logic for calculating metrics."""
        self.ind_u = 0
        """auxiliary variable so as not to lose the index in the distribution of resources."""
        self.window_size = 30

    def resAlloc(self, env):  # PEM ------------------------------------------------
        """This method implements Proportional Fair resource allocation between the different configured slices.
        This PEM method overwrites the resAlloc method from InterSliceScheduler class.

        Proportional Fair scheduler allocates all PRBs in the cell to the slice with the biggest metric.
        Metric for each slice is calculated as PossibleAverageUEtbs/ReceivedBytes."""
        while True:
            self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")
            if len(list(self.slices.keys())) > 0:
                if len(list(self.slices.keys())) > 1:
                    if env.now > 0:
                        first_array = []
                        second_array = []

                        for Slice in list(self.slices.keys()):
                            first_array.append(float(get_slice_throughput(self, Slice)))
                            second_array.append(get_slice_past_throughput(self, Slice))

                        assignSlicefactor(
                            self,
                            calculateSlicefactor(
                                self.scheduler, np.array((first_array, second_array))
                            ),
                        )
                        maxInd = findMaxMetSlice(self)
                        slice_prbs_allocate(self, maxInd)

                    else:
                        maxInd = random.choice(list(self.slices.keys()))
                        slice_prbs_allocate(self, maxInd)
                        self.printSliceConfig(maxInd)
                else:
                    slice = self.slices[list(self.slices.keys())[0]]
                    prbs = 0
                    for b in self.bw:
                        prbs = prbs + self.nRBtable[self.FR][slice.scs][str(b) + "MHz"]
                    slice.updateConfig(prbs)
                    self.printSliceConfig(slice.label)

            self.dbFile.write("<hr>")
            yield env.timeout(self.granularity)

    def printSliceConfig(self, slice):
        """This method stores inter slice scheduling debugging information on the log file, adding PF metric values."""
        super().printSliceConfig(slice)
        for s in list(self.slices.keys()):
            self.dbFile.write(
                "Slice: "
                + str(s)
                + " -> PF Metric: "
                + str(self.slices[s].metric)
                + "<br>"
            )



class PF2nd_Scheduler(InterSliceScheduler):
    """This class implements Proportional Fair inter slice scheduling algorithm."""

    def __init__(self, ba, fr, dm, tdd, gr, sch):
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)
        self.sch = sch
        """String formatted as PFXY, with X=numerator exponent for metric formula, and Y=denominator exponent. """
        self.rcvdBytesLen = 10
        """rcvdBytes list length in Slice instance. No more that rcvdBytesLen values are stored."""

    def resAlloc(self, env):  # PEM ------------------------------------------------
        """This method implements Proportional Fair resource allocation between the different configured slices.
        This PEM method overwrites the resAlloc method from InterSliceScheduler class.

        Proportional Fair scheduler allocates all PRBs in the cell to the slice with the biggest metric.
        Metric for each slice is calculated as PossibleAverageUEtbs/ReceivedBytes."""
        while True:
            self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")
            if len(list(self.slices.keys())) > 0:
                if len(list(self.slices.keys())) > 1:
                    if env.now > 0:
                        self.setMetric(float(self.sch[2]), float(self.sch[3]))
                        maxMetSlice = self.findMaxMetSlice()
                        self.assign2aSlice(maxMetSlice)
                        self.printSliceConfig(maxMetSlice)
                    else:
                        initialSlice = random.choice(list(self.slices.keys()))
                        self.assign2aSlice(initialSlice)
                        self.printSliceConfig(initialSlice)
                else:
                    slice = self.slices[list(self.slices.keys())[0]]
                    prbs = 0
                    for b in self.bw:
                        prbs = prbs + self.nRBtable[self.FR][slice.scs][str(b) + "MHz"]
                    slice.updateConfig(prbs)
                    self.printSliceConfig(slice.label)
            self.dbFile.write("<hr>")
            yield env.timeout(self.granularity)

    def setMetric(self, exp_n, exp_d):
        """This method sets the PF metric for each slice"""
        if len(list(self.slices.keys())) > 0:
            for slice in list(self.slices.keys()):
                rcvdBt_end = self.slices[slice].rcvdBytes[
                    len(self.slices[slice].rcvdBytes) - 1
                ]
                rcvdBt_in = self.slices[slice].rcvdBytes[0]
                if rcvdBt_end - rcvdBt_in == 0:
                    den = 1
                else:
                    den = rcvdBt_end - rcvdBt_in
                num = 0
                for ue in list(self.slices[slice].schedulerDL.ues.keys()):
                    [tbs, mod, bi, mcs] = self.slices[slice].schedulerDL.setMod(
                        ue, self.PRBs
                    )
                    num = num + tbs
                for ue in list(self.slices[slice].schedulerUL.ues.keys()):
                    [tbs, mod, bi, mcs] = self.slices[slice].schedulerUL.setMod(
                        ue, self.PRBs
                    )
                    num = num + tbs
                num = num / (
                    len(list(self.slices[slice].schedulerDL.ues.keys()))
                    + len(list(self.slices[slice].schedulerUL.ues.keys()))
                )
                self.slices[slice].metric = math.pow(float(num), exp_n) / math.pow(
                    den, exp_d
                )

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

    def assign2aSlice(self, slice):
        """This method allocates cell's PRBs to the indicated slice"""
        for sl in list(self.slices.keys()):
            if sl == slice:
                self.slices[sl].updateConfig(
                    int(self.PRBs / self.slices[sl].numRefFactor)
                )
            else:
                self.slices[sl].updateConfig(0)

    def printSliceConfig(self, slice):
        """This method stores inter slice scheduling debugging information on the log file, adding PF metric values."""
        super().printSliceConfig(slice)
        for s in list(self.slices.keys()):
            self.dbFile.write(
                "Slice: "
                + str(s)
                + " -> PF Metric: "
                + str(self.slices[s].metric)
                + "<br>"
            )



class QLearning_scheduler(InterSliceScheduler):
    """Subclass of InterSliceScheduler implementing a Q-Learning based scheduler.

    Attributes:
    - windows_size (int): The size of the sliding window for the Q-Learning algorithm.
    - learning_rate (float): The learning rate of the Q-Learning algorithm.
    - epsilon (float): The probability of selecting a random action for exploration.
    - epsilonDecay (float): The decay rate of epsilon for each iteration.
    - n_slices (int): The number of slices.
    - n_discrete_states (int): The number of discrete states per slice.
    - n_actions (int): The number of possible actions for the Q-Learning algorithm.
    - lowest_Q_value (float): The lowest value of the Q-Table.
    - highest_Q_value (float): The highest value of the Q-Table.
    - discount (float): The discount rate of the Q-Learning algorithm.
    - q_table (numpy.ndarray): The Q-Table used for the Q-Learning algorithm.
    - episode (int): The current episode of the Q-Learning algorithm.
    - discrete_state (list): The current discrete state of the Q-Learning algorithm.
    - new_discrete_state (list): The new discrete state of the Q-Learning algorithm.
    - SlicesWithPackets (int): The number of slices with packets in the Q-Learning algorithm.
    """
    def __init__(self, ba, fr, dm, tdd, gr):
        """Initializes a QLearning_scheduler object with the given parameters.

        Args:
        - ba (float): Bandwidth available for scheduling.
        - fr (float): Frequency at which the scheduler runs.
        - dm (int): Number of PRBs in the bandwidth.
        - tdd (bool): Boolean indicating whether Time Division Duplexing is used.
        - gr (bool): Boolean indicating whether to use group scheduling.
        """
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)

        self.windows_size = 30
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.epsilonDecay = 0.998
        self.n_slices = 2
        self.n_discrete_states = 20
        self.n_actions = 6
        self.lowest_Q_value = -2
        self.highest_Q_value = 0
        self.discount = 0.95

        self.q_table = np.random.uniform(
            low=self.lowest_Q_value,
            high=self.highest_Q_value,
            size=[self.n_discrete_states] + [self.n_discrete_states] + [self.n_actions],
        )
        self.episode = 0
        self.discrete_state = [0, 0]
        self.new_discrete_state = [0, 0]
        self.SlicesWithPackets = 0

    def get_reward(self):
        """Calculates and returns the reward for the current iteration of the Q-Learning algorithm.

        Returns:
        - The accumulated QoE.
        """
        accumulatedBits = 0
        accumulatedQoE = 0

        for slice in list(self.slices.keys()):
            accumulatedBits += self.slices[slice].schedulerDL.sliceBits
            accumulatedQoE += self.slices[slice].schedulerDL.sliceQoE

            self.slices[slice].schedulerDL.sliceBits = 0
            self.slices[slice].schedulerDL.sliceQoE = 0

        return accumulatedQoE
    def get_weights(self, action):
        """
        Returns the weights for the given action.

        Args:
            action (int): The action to get weights for.

        Returns:
            list: The weights for the given action.
        """
        if action == 0:
            return [0, 1]
        if action == 1:
            return [0.2, 0.8]
        if action == 2:
            return [0.4, 0.6]
        if action == 3:
            return [0.6, 0.4]
        if action == 4:
            return [0.8, 0.2]
        if action == 5:
            return []

    def resAlloc(self, env):
        """
        A generator function that performs resource allocation.

        Args:
            env: SimPy environment.
        """
        while True:
            if len(list(self.slices.keys())) == self.n_slices:
                self.slicesWithPackets = 0
                self.new_discrete_state = []

                # Perform Action
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.q_table[tuple(self.discrete_state)])
                else:
                    # Get random action
                    action = np.random.randint(0, self.n_actions)

                # Get weights for latter assignment
                weights = self.get_weights(action)

                # Perform assignment
                nSlice = 0
                for slice in list(self.slices.keys()):
                    self.slices[slice].updateConfig(
                        (
                            int(
                                (self.PRBs * weights[nSlice])
                                / self.slices[slice].numRefFactor
                            )
                        )
                    )
                    nSlice += 1

                # Get the system state
                for slice in list(self.slices.keys()):
                    if self.slices[slice].schedulerDL.updSumPcks() < 19:
                        self.new_discrete_state.append(
                            self.slices[slice].schedulerDL.updSumPcks()
                        )
                    else:
                        self.new_discrete_state.append(19)

                # Get reward, 1st is only considerates slice throughput
                reward = self.get_reward()

                # Q-learning update
                max_future_q = np.max(self.q_table[tuple(self.new_discrete_state)])
                current_q = self.q_table[tuple(self.discrete_state) + (action,)]

                new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                    reward + self.discount + max_future_q
                )

                # Update Q table with new Q value
                self.q_table[tuple(self.discrete_state) + (action,)] = new_q

                self.episode += 1

                # Assigns the new state to state, to perform corresponding action in next episode
                self.discrete_state = self.new_discrete_state

                self.epsilon *= self.epsilonDecay

            # If the number of slices is not ready yet, perform vanilla RR:
            else:
                if len(list(self.slices.keys())) > 1:
                    for slice in list(self.slices.keys()):
                        self.slices[slice].updateConfig(
                            int(
                                (self.PRBs / len(list(self.slices.keys())))
                                / self.slices[slice].numRefFactor
                            )
                        )
                        self.printSliceConfig(slice)
                else:
                    slice = self.slices[list(self.slices.keys())[0]]
                    prbs = 0
                    for b in self.bw:
                        prbs = prbs + self.nRBtable[self.FR][slice.scs][str(b) + "MHz"]
                    slice.updateConfig(prbs)

            yield env.timeout(self.granularity)


class debugging_scheduler(InterSliceScheduler):
    """This class implements Round Robin Plus inter slice scheduling algorithm."""

    def __init__(self, ba, fr, dm, tdd, gr):
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)
        self.k = 0

    def resAlloc(
        self, env, UEgroups, interSliceSche1, t_sim, bw, measInterv
    ):  # PEM ------------------------------------------------
        """This method implements Round Robin Plus PRB allocation between the different configured slices.
        This PEM method overwrites the resAlloc method from InterSliceScheduler class.

        Round Robin Plus scheduler allocates the same amount of resources to each slice with packets in buffer.
        """

        while True:
            self.k += 1
            self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")
            self.slicecounter = 0
            if len(list(self.slices.keys())) > 0:
                if len(list(self.slices.keys())) > 1:
                    print("debug")

                    state = np.zeros(len(list(self.slices.keys())), dtype=int)
                    i = 0
                    action = 0

                    slicesWithPacks = 0
                    for slice in list(self.slices.keys()):
                        sliceHasPackets = (
                            self.slices[slice].schedulerDL.updSumPcks() > 0
                            or self.slices[slice].schedulerUL.updSumPcks() > 0
                        )
                        print(self.slices[slice].schedulerDL.updSumPcks())
                        if sliceHasPackets:
                            slicesWithPacks = slicesWithPacks + 1
                        self.slicecounter += 1

                        state[i] = self.slices[slice].schedulerDL.updSumPcks() // 4
                        i += 1
                    print("selfslicecoutner: ", self.slicecounter)
                    if slicesWithPacks == 0:
                        for slice in list(self.slices.keys()):
                            self.slices[slice].updateConfig(
                                int(
                                    (self.PRBs / len(list(self.slices.keys())))
                                    / self.slices[slice].numRefFactor
                                )
                            )
                            self.printSliceConfig(slice)

                    else:
                        for slice in list(self.slices.keys()):
                            sliceHasPackets = (
                                self.slices[slice].schedulerDL.updSumPcks() > 0
                                or self.slices[slice].schedulerUL.updSumPcks() > 0
                            )
                            if not sliceHasPackets:
                                self.slices[slice].updateConfig(0)
                            else:
                                self.slices[slice].updateConfig(
                                    int(
                                        (self.PRBs / slicesWithPacks)
                                        / self.slices[slice].numRefFactor
                                    )
                                )
                            self.printSliceConfig(slice)
                        print(np.sort(state), state)

                    UEg = UEgroups[0]
                    if self.k > 3:
                        printResults(
                            "DL",
                            UEg.usersDL,
                            UEg.num_usersDL,
                            interSliceSche1.slices[UEg.label].schedulerDL,
                            t_sim,
                            True,
                            False,
                            UEg.sinr_0DL,
                        )
                else:
                    slice = self.slices[list(self.slices.keys())[0]]
                    prbs = 0
                    for b in self.bw:
                        prbs = prbs + self.nRBtable[self.FR][slice.scs][str(b) + "MHz"]
                    slice.updateConfig(prbs)
            self.dbFile.write("<hr>")
            yield env.timeout(self.granularity)


def printResults(
    dir, users, num_users, scheduler, t_sim, singleRunMode, fileSINR, sinr
):
    """This method prints main simulation results on the terminal"""
    PDRprom = 0.0
    SINRprom = 0.0
    MCSprom = 0.0
    THprom = 0.0
    # UEresults = open('UEresults'+str(sinr[0])+'.txt','w') # This file is used as an input for the validation script
    # UEresults.write('SINR MCS BLER PLR TH ResUse'+'\n')
    print(Format.CGREEN + "Accumulated " + dir + " indicators by user:" + Format.CEND)
    for i in range(num_users):
        # Count pending packets also as lost
        users[i].packetFlows[0].lostPackets = (
            users[i].packetFlows[0].lostPackets
            + len(list(scheduler.ues[users[i].id].pendingPckts.keys()))
            + len(users[i].packetFlows[0].appBuff.pckts)
            + len(scheduler.ues[users[i].id].bearers[0].buffer.pckts)
        )
        for p in list(scheduler.ues[users[i].id].pendingPckts.keys()):
            for pp in scheduler.ues[users[i].id].bearers[0].buffer.pckts:
                if p == pp.secNum:
                    users[i].packetFlows[0].lostPackets = (
                        users[i].packetFlows[0].lostPackets - 1
                    )

        users[i].packetFlows[0].setMeassures(t_sim)
        PDRprom = PDRprom + users[i].packetFlows[0].meassuredKPI["PacketLossRate"]
        THprom = THprom + users[i].packetFlows[0].meassuredKPI["Throughput"]
        if singleRunMode and fileSINR:
            sinrUser = float(users[i].radioLinks.lqAv) / users[i].radioLinks.totCount
        else:
            sinrUser = users[
                i
            ].radioLinks.linkQuality  # not single run, or single run but not taking SINR from file
        SINRprom = SINRprom + sinrUser
        MCSprom = MCSprom + float(users[i].MCS)

        print(
            users[i].id
            + "\t"
            + Format.CYELLOW
            + " Sent Packets:"
            + Format.CEND
            + str(users[i].packetFlows[0].sentPackets)
            + Format.CYELLOW
            + " Lost Packets:"
            + Format.CEND
            + str(users[i].packetFlows[0].lostPackets)
        )
        print(
            "\t"
            + "SINRav: "
            + str(int(sinrUser))
            + " MCSav: "
            + str(users[i].MCS)
            + " PLR: "
            + str(round(users[i].packetFlows[0].meassuredKPI["PacketLossRate"], 2))
            + " %"
            + " Throughput: "
            + str(round(users[i].packetFlows[0].meassuredKPI["Throughput"], 2))
        )

        # UEresults.write(str(sinrUser)+' '+str(users[i].MCS)+' '+str(float(users[i].lostTB)/(users[i].TXedTB+users[i].lostTB))+' '+str(users[i].packetFlows[0].meassuredKPI['PacketLossRate']/100)+' '+str(users[i].packetFlows[0].meassuredKPI['Throughput'])+' '+str(users[i].resUse)+' '+str(int(float(users[i].tbsz)/8))+'\n')

    print(Format.CGREEN + "Average " + dir + " Indicators:" + Format.CEND)
    print("Packet Loss Rate av: " + "\t" + str(round((PDRprom) / num_users, 2)) + " %")
    print("Throughput av: " + "\t" + str(round(THprom / num_users, 2)) + " Mbps")
    print("Connections av: " + "\t" + str(num_users))
    print("Slice Resources: " + "\t" + str(scheduler.nrbUEmax) + " PRBs")
    print("Symbols in slot: " + "\t" + str(scheduler.TDDsmb))
    print("Slice Numerology: " + "\t" + str(scheduler.ttiByms * 15) + " kHz")