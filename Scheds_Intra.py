"""This module contains different implemented intra slice schedulers.
New schedulers should be implemented here following the current structure."""

import math
import pickle
import sys
from collections import deque

import numpy as np

from IntraSliceSch import Format, IntraSliceScheduler

sys.path.insert(1, "lib")
from lib.AISchedulersUtils import *
from lib.SchedulerUtils import *
from lib.uestats import past_throughputs_update, get_throughput, average_past_throughputs, get_delay


# ROUND ROBIN SCHEDULER #

class RR_Scheduler(IntraSliceScheduler):
    def __init__(
        self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch
    ):
        """
        Constructor for the RR_Scheduler class.

        :param ba: Bandwidth allocation for the slice
        :type ba: int
        :param n: Number of symbols in a slot
        :type n: int
        :param debMd: Debugging mode flag
        :type debMd: bool
        :param sLod: Slice load for the slice
        :type sLod: float
        :param ttiByms: Time interval for a TTI in milliseconds
        :type ttiByms: float
        :param mmd_: Mode flag for scheduler
        :type mmd_: bool
        :param ly_: Layer for the scheduler
        :type ly_: int
        :param dir: Direction of the scheduler
        :type dir: str
        :param Smb: Size of the MAC buffer for the slice
        :type Smb: int
        :param robustMCS: MCS value for robustness
        :type robustMCS: int
        :param slcLbl: Label for the slice
        :type slcLbl: str
        :param sch: Scheduler type
        :type sch: str
        """
        IntraSliceScheduler.__init__(
            self,
            ba,
            n,
            debMd,
            sLod,
            ttiByms,
            mmd_,
            ly_,
            dir,
            Smb,
            robustMCS,
            slcLbl,
            sch,
        )
        self.schType = sch
        self.PRBs_used = 0

    def resAlloc(self, Nrb):
        """
        This method allocates cell PRBs to the different connected UEs.

        :param Nrb: Number of PRBs to allocate
        :type Nrb: int
        :return: Number of PRBs assigned to UEs
        :rtype: int
        """
        schd = self.schType[0:2]

        rb_assigned = 0

        if schd == "Rr" and len(list(self.ues.keys())) > 0:
            n_uesWithPackets = 0
            for ue in list(self.ues.keys()):
                if len(self.ues[ue].bearers[0].buffer.pckts) > 0:
                    n_uesWithPackets += 1

            n_uesWithPackets = 1 if n_uesWithPackets == 0 else n_uesWithPackets
            prbs_per_ue = math.floor(Nrb / n_uesWithPackets)
            sobrantes = 0
            if prbs_per_ue != 0:
                sobrantes = Nrb % prbs_per_ue
            cont = 0
            buffer = Nrb

            for ue in list(
                self.ues.keys()
            ):
                #print('delay: ', get_delay(self.ues[ue]))
                if (
                    len(self.ues[ue].bearers[0].buffer.pckts) > 0
                    and n_uesWithPackets > 0
                ):
                    cont+=1
                    if cont != n_uesWithPackets:
                        if self.ues[ue].BWPs < 37:
                            assignation = min(prbs_per_ue,4,buffer) 
                            self.ues[ue].prbs = assignation
                            buffer -= assignation
                            rb_assigned += assignation

                        elif self.ues[ue].BWPs < 73:
                            assignation = min(prbs_per_ue,8,buffer) 
                            self.ues[ue].prbs = assignation
                            buffer -= assignation
                            rb_assigned += assignation
                        else:
                            assignation = min(prbs_per_ue,buffer) 
                            self.ues[ue].prbs = assignation
                            buffer -= assignation
                            rb_assigned += assignation
                            
                    else:
                        self.ues[ue].prbs = max(sobrantes, buffer)
                        rb_assigned += max(sobrantes, buffer)
            
        self.printResAlloc()
        
        return rb_assigned
    


# PROPORTIONAL FAIR SCHEDULER #

class PF_scheduler(IntraSliceScheduler):
    """
    A class that implements Proportional Fair intra-slice scheduling algorithm.

    Attributes:
    - promLen (int): Length of the past throughput average considered in PF metric.
    - scheduler (framework_UEfactor): Object of type framework_UEfactor that contains the logic for calculating metrics.
    - ind_u (int): Auxiliary variable so as not to lose the index in the distribution of resources.
    - window_size (int): Window size for resource allocation.

    """

    def __init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch):
        """
        Initializes the Pp_scheduler instance.

        Args:
        - ba (int): Bandwidth.
        - n (int): Number of subcarriers.
        - debMd (bool): Debug mode.
        - sLod (int): Number of slices.
        - ttiByms (int): Transmission time interval.
        - mmd_ (int): Modulation order.
        - ly_ (int): Number of layers.
        - dir (str): Directory.
        - Smb (int): Maximum number of bits that can be transmitted in one subframe.
        - robustMCS (bool): Robust MCS.
        - slcLbl (int): Slice label.
        - sch (int): Scheduling.

        """
        
        IntraSliceScheduler.__init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch)
        self.promLen = 30
        self.scheduler = framework_UEfactor([layer("P", np.array([float(self.schType[2]), -float(self.schType[3])]), "id")])
        self.ind_u = 0
        self.window_size = 30

    def resAlloc(self, band):
        """
        Implements Proportional Fair resource allocation between the different connected UEs.
        This method overwrites the resAlloc method from IntraSliceScheduler class.

        Proportional Fair scheduler allocates all PRBs in the slice to the UE with the biggest metric.
        Metric for each UE is calculated as PossibleUEtbs/AveragePastTbs.

        Args:
        - band (int): Band.

        """
        schd = self.schType[0:2]
        if schd == "PF" and len(list(self.ues.keys())) > 0:
            num_array = []  
            den_array = []  

            # Collect throughput data for each UE
            for ue in list(self.ues.keys()):
                num_array.append(float(get_throughput(self, ue)))
                den_array.append(average_past_throughputs(self, ue))

            # Calculate metric using the framework_UEfactor object
            assignUEfactor(self, calculateUEfactor(self.scheduler, np.array((num_array, den_array))))

            # Find the index of the UE with the highest metric
            maxInd, self.ind_u = findMaxFactor(self, self.ind_u)

            # Allocate resources to the UE with the highest metric
            prbs_allocate(self, maxInd, self.window_size, band)

            # Print Resource Allocation
            self.printResAlloc()
            
            return band


# ROUND ROBIN PLUS SCHEDULER #

class RRp_Scheduler(IntraSliceScheduler):
    def __init__(
        self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch
    ):
        """Construct a new RRp_Scheduler instance. 

        Args:
            ba (int): Bandwidth
            n (int): Sub-carrier spacing
            debMd (int): Debugging mode
            sLod (int): Sub-loads
            ttiByms (int): TTI in ms
            mmd_ (str): Modulation and coding scheme
            ly_ (str): Layer type
            dir (str): Direction
            Smb (int): Maximum buffer size
            robustMCS (bool): Robust MCS
            slcLbl (str): Slice label
            sch (str): Scheduler type
        """
        IntraSliceScheduler.__init__(
            self,
            ba,
            n,
            debMd,
            sLod,
            ttiByms,
            mmd_,
            ly_,
            dir,
            Smb,
            robustMCS,
            slcLbl,
            sch,
        )
        self.schType = sch

    def resAlloc(self, Nrb):
        """Allocate cell PRBs to the different connected UEs.

        Args:
            Nrb (int): Number of Resource Blocks

        """
        schd = self.schType[0:2]
        if schd == "RR" and len(list(self.ues.keys())) > 0:
            n_uesWithPackets = uesWithPackets(self)
            for ue in list(self.ues.keys()):
                if len(self.ues[ue].bearers[0].buffer.pckts) > 0:
                    n_uesWithPackets += 1
            for ue in list(
                self.ues.keys()
            ):  # TS 38.214 table 5.1.2.2.1-1  RBG size for configuration 2
                if (
                    len(self.ues[ue].bearers[0].buffer.pckts) > 0
                    and n_uesWithPackets > 0
                ):
                    self.ues[ue].prbs = math.floor(
                        Nrb / n_uesWithPackets
                    )  # len(list(self.ues.keys()))) # To compare with lena-5G
                else:
                    self.ues[ue].prbs = 0

        # Print Resource Allocation
        self.printResAlloc()



# 2nd PROPORTIOANL FAIR SCHEDULER #

class PF2nd_Scheduler(IntraSliceScheduler):
    """
    This class implements Proportional Fair intra slice scheduling algorithm.
    """

    def __init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch):
        """
        Constructor method for PF2nd_Scheduler class.

        :param ba: bandwidth.
        :param n: number of PRBs.
        :param debMd: debug mode.
        :param sLod: slice load.
        :param ttiByms: TTI in ms.
        :param mmd_: mobility mode.
        :param ly_: slice layer.
        :param dir: direction.
        :param Smb: SISO MIMO bit capacity.
        :param robustMCS: robust MCS.
        :param slcLbl: slice label.
        :param sch: scheduling algorithm.
        """
        IntraSliceScheduler.__init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch)
        self.promLen = 30  # Past Throughput average length considered in PF metric

    def resAlloc(self, band):
        """
        This method implements Proportional Fair resource allocation between the different connected UEs.
        This method overwrites the resAlloc method from IntraSliceScheduler class.

        Proportional Fair scheduler allocates all PRBs in the slice to the UE with the biggest metric.
        Metric for each UE is calculated as PossibleUEtbs/AveragePastTbs.

        :param band: number of PRBs.
        """

        schd = self.schType[0:2]
        if schd == 'PF' and len(list(self.ues.keys())) > 0:
            exp_num = float(self.schType[2])
            exp_den = float(self.schType[3])
            self.setUEfactor(exp_num, exp_den)
            maxInd = self.findMaxFactor()
            for ue in list(self.ues.keys()):
                if ue == maxInd:
                    self.ues[ue].prbs = band
                else:
                    self.ues[ue].prbs = 0
                    if len(self.ues[ue].pastTbsz) > self.promLen:
                        self.ues[ue].pastTbsz.popleft()
                    self.ues[ue].pastTbsz.append(self.ues[ue].tbsz)
                    self.ues[ue].tbsz = 1
        # Print Resource Allocation
        self.printResAlloc()
        return band

    def setUEfactor(self, exp_n, exp_d):
        """
        This method sets the PF metric for each UE.

        :param exp_n: numerator exponent.
        :param exp_d: denominator exponent.
        """
        for ue in list(self.ues.keys()):
            sumTBS = 0
            for t in self.ues[ue].pastTbsz:
                sumTBS = sumTBS + t
            actual_den = sumTBS / len(self.ues[ue].pastTbsz)
            [tbs, mod, bi, mcs] = self.setMod(ue, self.nrbUEmax)
            self.ues[ue].pfFactor = math.pow(float(tbs), exp_n) / math.pow(actual_den, exp_d)
            self.ues[ue].lastDen = actual_den
            self.ues[ue].num = tbs

    def findMaxFactor(self):
        """This method finds and returns the UE with the highest metric"""
        factorMax = 0
        factorMaxInd = ''
        for ue in list(self.ues.keys()):
            if len(self.ues[ue].bearers[0].buffer.pckts)>0 and self.ues[ue].pfFactor>factorMax:
                factorMax = self.ues[ue].pfFactor
                factorMaxInd = ue
        if factorMaxInd=='':
            ue = list(self.ues.keys())[self.ind_u]
            q = 0
            while len(self.ues[ue].bearers[0].buffer.pckts)==0 and q<len(self.ues):
                self.updIndUE()
                ue = list(self.ues.keys())[self.ind_u]
                q = q + 1
            factorMaxInd = ue

        return factorMaxInd

    def printResAlloc(self):
        if self.dbMd:
            self.printDebData('+++++++++++ Res Alloc +++++++++++++'+'<br>')
            self.printDebData('PRBs: '+str(self.nrbUEmax)+'<br>')
            resAllocMsg = ''
            for ue in list(self.ues.keys()):
                resAllocMsg = resAllocMsg + ue +' '+ str(self.ues[ue].pfFactor)+' '+str(self.ues[ue].prbs)+ ' '+str(self.ues[ue].num)+' '+ str(self.ues[ue].lastDen)+'<br>'
            self.printDebData(resAllocMsg)
            self.printDebData('+++++++++++++++++++++++++++++++++++'+'<br>')


# SVM SCHEDULER #

class SV_Scheduler(IntraSliceScheduler):
    """
    A class that extends the IntraSliceScheduler class for implementing a resource allocator algorithm.

    Inherits:
        IntraSliceScheduler: a class for scheduling network resources for connected UEs.
    """
def __init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch):
    """
    Initializes an object of the SV_Scheduler class.

    Args:
        ba (int): Base station number.
        n (int): Number of PRBs in the slice.
        debMd (bool): Flag for debug mode.
        sLod (int): Total load in the slice.
        ttiByms (int): TTI length in milliseconds.
        mmd_ (int): Max mcs index for dl.
        ly_ (str): Type of layer to use.
        dir (str): Directory for storing data.
        Smb (int): Initial number of sub-bands.
        robustMCS (bool): Robust MCS flag.
        slcLbl (int): Slice label.
        sch (str): Scheduler type.

    """

    # call the constructor of the IntraSliceScheduler class:
    IntraSliceScheduler.__init__(self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch)

    # create a layer of calculations to be carried out:
    exp_num = float(self.schType[2])
    exp_den = float(self.schType[3])
    layer1 = layer("P", np.array([exp_num, -exp_den]), "id")

    # create an object of type framework_UEfactor that contains the logic for calculating metrics:
    self.scheduler = framework_UEfactor([layer1])

    # create an auxiliary variable so as not to lose the index in the distribution of resources:
    self.ind_u = 0

    # set the promotion length and window size:
    self.promLen = 100
    self.window_size = self.promLen

    # set the time at which prediction is to be done:
    self.predict_at = int(self.schType[3:5])

    # create an object that carries the necessary methods for the development of a supervised resource allocator algorithm:
    self.svm_scheduler = supervised_scheduler(self.predict_at, self.scheduler)

def resAlloc(self, band):
    """
    This method implements a resource allocation between the different connected UEs.

    This method overwrites the resAlloc method from the IntraSliceScheduler class.

    Proportional Fair scheduler allocates all PRBs in the slice to the UE with the biggest metric.
    In this method, an algorithm is developed through which the assignment is learned
    based on the experience of a proportional fair based allocator.

    Args:
        band (int): The band to allocate resources in.

    """

    # get the scheduler type:
    schd = self.schType[0:2]

    # check if there are any UEs connected:
    if schd == "SV" and len(list(self.ues.keys())) > 0:

        # create an array that carries the throughput vector for the TTI:
        num_array = []

        # create an array that carries the vector of throughput averaged at the moment:
        den_array = []

        for ue in list(self.ues.keys()):
            num_array.append(get_throughput(self, ue, self.nrbUEmax))
            den_array.append(average_past_throughputs(self.ues[ue]))

        # feature vector used for learning:
        self.svm_scheduler.characteristics = np.array([num_array, den_array])

        # once the critical episode is reached, the vector of characteristics and
        # labels is opened to learn the model.
        if self.svm_scheduler.episode == self.svm_scheduler.predicting_episode:
            with open("data/data_vectors", "rb") as fr:
                try:
                    while True:
                        self.svm_scheduler.x_array.append(pickle.load(fr))
                except EOFError:
                    pass
            with open("data/data_labels", "rb") as fr2:
                try:
                    while True:
                        self.svm_scheduler.y_array.append(pickle.load(fr2))
                except EOFError:
                    pass

            self.svm_scheduler.mlmodel = SVC(kernel="linear")
            self.svm_scheduler.mlmodel.fit(
                self.svm_scheduler.x_array, self.svm_scheduler.y_array
            )

            # find the index of the list of ues with the highest metric using the sv model:
            maxInd = self.svm_scheduler.get_index(self.ues)

            # assign the resources to the user with maxInd:
            prbs_allocate(maxInd, self.ues, self.window_size, band)

        # Print Resource Allocation
        self.printResAlloc()



# Q-LEARNING SCHEDULER #

class QL_Scheduler(IntraSliceScheduler):
    def __init__(
        self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch
    ):
        IntraSliceScheduler.__init__(
            self,
            ba,
            n,
            debMd,
            sLod,
            ttiByms,
            mmd_,
            ly_,
            dir,
            Smb,
            robustMCS,
            slcLbl,
            sch,
        )

        self.num_states = 10
        """number of possible states."""
        self.num_actions = 2
        """number of possible actions."""
        self.qmodel = qlearning_scheduler(self.num_states, self.num_actions)
        """object that has the necessary tools to develop the q learning algorithm"""
        self.action = 0
        """action to be taken"""
        self.eps = 0.3
        """value between 0 and 1, represents how likely I will perform a random action 
        or follow the policy obtained."""
        self.contador = 0
        """used to avoid the first assignment cases (ambiguous)"""
        self.promLen = 30
        """Past Throughput average length"""
        self.reward = 0
        """reward for having performed the action."""
        self.window_size = self.promLen
        self.conversion_factor = 10 / 250000
        """scale factor to work with the metric range"""

    def set_state(self, ues):
        """Returns an array with the current state of the system, in this case the array
        corresponds to the metric of the proportional fair for each UE, scaled."""
        messy_states = np.zeros(len(list(ues.keys())), dtype=int)
        i = 0
        for ue in list(ues.keys()):
            ues[ue].pfFactor = get_throughput(
                self, ue, self.nrbUEmax
            ) / average_past_throughputs(ues[ue])
            messy_states[i] = int(ues[ue].pfFactor * self.conversion_factor)
            print("pffactor: ", ues[ue].pfFactor)
            i += 1

        return messy_states

    def resAlloc(self, band):
        """This method implements a resource allocation between the different connected UEs.
        This method overwrites the resAlloc method from IntraSliceScheduler class.

        This method develops a resource allocation algorithm based on Q-learning. First the
        current state is built, then an action is performed based on the greedy policy,
        then the new state is taken after having performed the action. With all this information,
        table Q is updated."""
        schd = self.schType[0:2]

        if schd == "QL" and len(list(self.ues.keys())) > 0:
            if (
                self.contador > 20
            ):  # used to avoid the first assignment cases (ambiguous)
                self.state = np.zeros(len(list(self.ues.keys())), dtype=int)

                # get the states ordered from lowest to highest to facilitate learning:
                self.state = sorted(self.set_state(self.ues))

                # I learn the action based on the greedy policy, it represents which user
                # to assign the resources to:
                self.action = self.qmodel.EpsilonGreedyPolicy(self.eps, self.state)

                # I need the user from the unordered list:
                for ue in list(self.ues.keys()):
                    if (int(self.ues[ue].pfFactor * self.conversion_factor)) == (
                        self.state[self.action - 1]
                    ):
                        maxInd = ue

                # assign the resources to the user with maxInd:
                prbs_allocate(maxInd, self.ues, self.window_size, band)

                # get the new state:
                self.next_state = sorted(self.set_state(self.ues))

                # get the reward:
                for ue in list(self.ues.keys()):
                    self.reward += int(
                        self.ues[maxInd].pfFactor * self.conversion_factor
                    )

                # update the q table with the above information:
                self.qmodel.update(
                    self.state, self.action, self.reward, self.next_state, 0
                )

            else:
                prbs_allocate(
                    list(self.ues.keys())[0], self.ues, self.window_size, band
                )

            self.contador = self.contador + 1

        # Print Resource Allocation
        self.printResAlloc()
        return band


# 2nd Q-LEARNING SCHEDULER #

class QL2d_Scheduler(IntraSliceScheduler): 
    """This class implements Q-Learning intra slice scheduling algorithm."""

    def __init__(
        self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch
    ):
        IntraSliceScheduler.__init__(
            self,
            ba,
            n,
            debMd,
            sLod,
            ttiByms,
            mmd_,
            ly_,
            dir,
            Smb,
            robustMCS,
            slcLbl,
            sch,
        )
        self.promLen = 30

        """auxiliary variable so as not to lose the index in the distribution of resources."""
        self.window_size = 30
        self.LEARNING_RATE = 0.1
        self.eps = 0.8

        self.q_table = np.random.uniform(low=-2, high=0, size=[2] + [10, 10] + [4])
        self.win_throughput = 220000 / 10
        self.win_avgthroughput = 220000 / 10
        self.ep = 0
        self.state = [0, 0, 0]
        self.nstate = [0, 0, 0]
        self.action = 0

    def resAlloc(self, band):
        """This method implements Proportional Fair resource allocation between the different connected UEs.
        This method overwrites the resAlloc method from IntraSliceScheduler class.

        Proportional Fair scheduler allocates all PRBs in the slice to the UE with the biggest metric.
        Metric for each UE is calculated as PossibleUEtbs/AveragePastTbs."""
        schd = self.schType[0:3]
        if schd == "Q2" and len(list(self.ues.keys())) > 0 and self.ep > 5:
            num_array = []  # array that carries the throughput vector for the TTI
            den_array = (
                []
            )  # array that carries the vector of throughput averaged at the moment
            ue_array = []

            print(np.shape(self.nstate))
            print(np.shape(self.q_table))

            max_future_q = np.max(self.q_table[self.nstate])

            # 221832         220000 / 10
            for ue in list(self.ues.keys()):
                num_array.append(
                    (float(get_throughput(self, ue))) // self.win_throughput
                )
                den_array.append(
                    (average_past_throughputs(self, ue)) // self.win_avgthroughput
                )
                # ue_array.append(int(ue[2])-1)

            # state = np.array([num_array*win_throughput, den_array*win_avg_throughput])
            self.nstate[0] = [num_array[0], den_array[0]]
            self.nstate[1] = [num_array[1], den_array[1]]

            current_q = self.q_table[self.state + (self.action,)]

            if np.random.random() > eps:
                self.action = np.argmax(self.q_table[self.nstate])
            else:
                self.action = np.random.randint(0, self.max_action)

            reward = den_array[0] + den_array[1]

            new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (
                reward + self.DISCOUNT * max_future_q
            )

            self.q_table[self.state + (self.action,)] = new_q

            # assign the resources to the user

            # for ue in list(self.ues.keys()):
            if action == 0:
                self.ues["ue0"] = np.floor(band)
                self.ues["ue1"] = 0
            if action == 1:
                self.ues["ue0"] = np.floor(0.7 * band)
                self.ues["ue1"] = np.floor(0.3 * band)
            if action == 2:
                self.ues["ue0"] = np.floor(0.3 * band)
                self.ues["ue1"] = np.floor(0.7 * band)
            if action == 3:
                self.ues["ue0"] = 0
                self.ues["ue1"] = np.floor(band)

            for ue in list(self.ues.keys()):
                past_throughputs_update(self.ues[ue], window_size)

            self.state = self.nstate

        else:
            if self.ep <= 5:
                for ue in list(self.ues.keys()):
                    self.ues[ue].prbs = 1

        # Print Resource Allocation
        self.printResAlloc()

        self.ep += 1

    def get_action(self, action):
        if action == 0:
            return [0, 1]
        elif action == 1:
            return [0.3, 0.7]
        elif action == 2:
            return [0.7, 0.3]
        elif action == 3:
            return [0, 1]

    def printResAlloc(self):
        if self.dbMd:
            self.printDebData("+++++++++++ Res Alloc +++++++++++++" + "<br>")
            self.printDebData("PRBs: " + str(self.nrbUEmax) + "<br>")
            resAllocMsg = ""
            for ue in list(self.ues.keys()):
                resAllocMsg = (
                    resAllocMsg
                    + ue
                    + " "
                    + str(self.ues[ue].pfFactor)
                    + " "
                    + str(self.ues[ue].prbs)
                    + " "
                    + str(self.ues[ue].num)
                    + " "
                    + str(self.ues[ue].lastDen)
                    + "<br>"
                )
            self.printDebData(resAllocMsg)
            self.printDebData("+++++++++++++++++++++++++++++++++++" + "<br>")

            

# TDD SCHEDULER #

class TDD_Scheduler(IntraSliceScheduler):  # TDD Sched ---------
    """This class implements TDD intra slice scheduling."""

    def __init__(
        self, ba, n, debMd, sLod, ttiByms, mmd_, ly_, dir, Smb, robustMCS, slcLbl, sch
    ):
        IntraSliceScheduler.__init__(
            self,
            ba,
            n,
            debMd,
            sLod,
            ttiByms,
            mmd_,
            ly_,
            dir,
            Smb,
            robustMCS,
            slcLbl,
            sch,
        )
        self.symMax = Smb
        self.queue = TBqueueTDD(self.symMax)
        """TDD scheduler TB queue.

        IntraSliceScheduler class attribute queue is overwriten here by a new type of queue
        which handles symbols. This queue will contain as much TB as a slot can contain. If resource allocation is made
        in terms of slots, it will contain 1 element, else, it will contain as much mini-slots as can be supported in 1 slot."""

    def resAlloc(self, band):
        """This method implements resource allocation between the different connected UEs in a TDD slice.

        It overwrites the resAlloc method from IntraSliceScheduler class.
        In this Py5cheSim version TDD scheduler allocates all PRBs in the slice to a UE during 1 slot.
        Future Py5cheSim versions could support mini-slot allocation by changing the UE symbol allocation in this method.
        Note that in that case, althoug there is no need to update the queueUpdate method,
        TBS calculation must be adjusted to avoid losing capacity when trunking the Nre__ value.
        """

        if len(list(self.ues.keys())) > 0:
            for ue in list(self.ues.keys()):
                self.ues[ue].prbs = band
                self.ues[ue].symb = self.TDDsmb
        # Print Resource Allocation
        self.printResAlloc()

    def queueUpdate(self):
        """This method fills scheduler TB queue at each TTI with TBs built with UE data/signalling bytes.

        It overwrites queueUpdate method from IntraSliceScheduler class, making Resource allocation in terms of slot Symbols
        and insert generated TBs into Scheduler queue in a TTI. Althoug in this version Resource allocation is made by slot,
        it is prepared to support mini-slot resource allocation by handling a scheduler TB queue in terms of symbols.
        """
        packts = 1
        self.ueLst = list(self.ues.keys())
        self.resAlloc(self.nrbUEmax)
        sym = 0
        if self.nrbUEmax == 0:
            self.sm_lim = 0
        else:
            if self.mimomd == "MU":
                self.sm_lim = self.symMax * self.nlayers
            else:
                self.sm_lim = self.symMax

        while len(self.ueLst) > 0 and packts > 0 and sym < self.sm_lim:
            ue = self.ueLst[self.ind_u]
            self.printDebDataDM(
                "---------------- " + ue + " ------------------<br>"
            )  # print more info in debbug mode
            if self.ues[ue].symb > 0:
                if len(self.ues[ue].bearers) > 0 and sym < self.sm_lim:
                    if len(self.ues[ue].pendingTB) == 0:  # No TB to reTX
                        sym = sym + self.rrcUncstSigIn(ue)
                        if (
                            sym < self.sm_lim
                            and len(self.ues[ue].bearers[0].buffer.pckts) > 0
                        ):
                            sym = sym + self.dataPtoTB(ue)
                    else:  # There are TB to reTX
                        self.printPendTB()
                        sym = sym + self.retransmitTB(ue)
                    if self.dbMd:
                        self.printQtb()  # Print TB queue in debbug mode
            self.updIndUE()
            packts = self.updSumPcks()

    def rrcUncstSigIn(self, u):
        ueN = int(self.ues[u].id[2:])
        sfSig = int(float(1) / self.sLoad)
        rrcUESigCond = (self.sbFrNum - ueN) % sfSig == 0
        if rrcUESigCond:
            p_l = []
            p_l.append(self.ues[u].packetFlows[0].pId)
            self.ues[u].packetFlows[0].pId = self.ues[u].packetFlows[0].pId + 1
            ins = self.insertTB(
                self.ues[u].TBid, "4-QAM", u, "Sig", p_l, self.ues[u].prbs, 19
            )
            r = self.symMax
        else:
            r = 0
        return r

    def retransmitTB(self, u):
        pendingTbl = self.ues[u].pendingTB[0]
        if pendingTbl.reTxNum < 3000:  # TB retransmission
            intd = self.queue.insertTB(pendingTbl)
            self.ues[u].pendingTB.pop(0)
            pendingTbl.reTxNum = pendingTbl.reTxNum + 1
            r = self.symMax
        else:
            self.ues[u].pendingTB.pop(0)  # Drop!!!
            r = 0
        return r

    def dataPtoTB(self, u):
        """This method takes UE data bytes, builds TB and puts them in the scheduler TB queue.

        It overwrites dataPtoTB method from IntraSliceScheduler class. In this case it returns
        the amount of allocated symbols to the UE."""
        n = self.ues[u].prbs
        [tbSbits, mod, bits, mcs__] = self.setMod(u, n)
        if self.schType[0:2] == "PF":
            if len(self.ues[u].pastTbsz) > self.promLen:
                self.ues[u].pastTbsz.popleft()
            self.ues[u].pastTbsz.append(self.ues[u].tbsz)

        self.ues[u].tbsz = tbSbits
        self.ues[u].MCS = mcs__
        self.setBLER(u)
        tbSize = int(float(tbSbits) / 8)  # TB size in bytes
        self.printDebDataDM(
            "TBs: "
            + str(tbSize)
            + " nrb: "
            + str(n)
            + " FreeSp: "
            + str(self.queue.getFreeSpace())
            + "<br>"
        )
        pks_s = 0
        list_p = []
        while pks_s < tbSize and len(self.ues[u].bearers[0].buffer.pckts) > 0:
            pacD = self.ues[u].bearers[0].buffer.removePckt()
            pks_s = pks_s + pacD.size
            list_p.append(pacD.secNum)

        insrt = self.insertTB(
            self.ues[u].TBid, mod, u, "data", list_p, n, min(int(pks_s), tbSize)
        )
        if (pks_s - tbSize) > 0:
            pacD.size = pks_s - tbSize
            self.ues[u].bearers[0].buffer.insertPcktLeft(pacD)
        return self.ues[u].symb

    def setTBS(self, r, qm, uldl, u_, fr, nprb):  # TS 38.214 procedure
        OHtable = {"DL": {"FR1": 0.14, "FR2": 0.18}, "UL": {"FR1": 0.08, "FR2": 0.10}}
        OH = OHtable[uldl][fr]
        Nre__ = min(156, math.floor(12 * self.ues[u_].symb * (1 - OH)))
        if self.mimomd == "SU":
            Ninfo = Nre__ * nprb * r * qm * self.nlayers
            tbs = Ninfo
        else:
            Ninfo = Nre__ * nprb * r * qm
            tbs = Ninfo
        return tbs

    def printResAlloc(self):
        if self.dbMd:
            self.printDebData("+++++++++++ Res Alloc +++++++++++++" + "<br>")
            self.printDebData("PRBs: " + str(self.nrbUEmax) + "<br>")
            resAllocMsg = ""
            for ue in list(self.ues.keys()):
                resAllocMsg = (
                    resAllocMsg
                    + ue
                    + ": "
                    + str(self.ues[ue].symb)
                    + " symbols"
                    + "<br>"
                )
            self.printDebData(resAllocMsg)
            self.printDebData("+++++++++++++++++++++++++++++++++++" + "<br>")


class TBqueueTDD:  # TB queue!!!
    """This class is used to model scheduler TB queue in TDD scheduler."""

    def __init__(self, symb):
        self.res = deque([])
        self.numRes = symb

    def getFreeSpace(self):
        freeSpace = self.numRes
        if len(self.res) > 0:
            for tbl in self.res:
                freeSpace = freeSpace - 1
        return freeSpace

    def insertTB(self, tb):
        succ = False
        freeSpace = self.getFreeSpace()
        if freeSpace >= 1:
            self.res.append(tb)  # The TB fits the free space
            succ = True
        else:
            succ = False
            print(
                Format.CRED
                + "Not enough space!!! : "
                + str(freeSpace)
                + "/"
                + str(tb.numRB)
                + Format.CEND
            )
        return succ

    def removeTB(self):
        if len(self.res) > 0:
            return self.res.popleft()

    def updateSize(self, newSize):
        self.numRes = newSize
