"""This module contains the Slice class.
"""
from collections import deque

from IntraSliceSch import IntraSliceScheduler, LTE_scheduler
from Scheds_Intra import *


class Slice:
    """This class has Slice relative parameters and is used to implement the mapping between service requirements and slice configuration."""

    def __init__(
        self, dly, thDL, thUL, avl, cnxDL, cnxUL, ba, dm, mmd, ly, lbl, tdd, sch
    ):
        self.reqDelay = dly
        self.reqThroughputDL = thDL
        self.reqThroughputUL = thUL
        self.reqAvailability = avl
        self.reqDLconnections = cnxDL
        self.reqULconnections = cnxUL

        self.band = ba
        self.PRBs = 0
        self.signLoad = 0.000001
        self.scs = "15kHz"
        self.ttiBms = 1
        self.robustMCS = False
        self.mimoMd = mmd
        self.layers = ly
        self.tddSymb = 14
        self.schType = sch
        self.label = lbl
        self.tdd = tdd
        self.dm = dm
        self.rcvdBytes = deque([0])  # interSlice PF scheduler
        self.metric = 0  # interSlice PF scheduler
        self.setInitialConfig()
        self.schedulerDL = self.createSliceSched("DL", self.tddSymb)
        if self.label != "LTE":
            self.schedulerUL = self.createSliceSched("UL", 14 - self.tddSymb)

        self.nUsers = 0
        self.metric = 0  # feb 2022
        self.reqThroughput = thDL*0.8 # 15#thDL # A definir
        if self.label == 'URLLC':
            self.reqThroughput = 7.5
        if self.label == 'VoLTE':
            self.reqThroughput = 10.9
        if self.label == 'Video':
            self.reqThroughput = 5
        self.state = []
        self.reward = 0


        

        self.buffer_available = False

    def createSliceSched(self, dir, tddSymb):
        """This method initializes and returns slice UL or DL scheduler. Scheduler algorithm is selected according to the Slice attribute schType."""
        if self.tdd:
            if self.schType[0:2] == "PF":
                scheduler = PF_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    tddSymb,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            else:
                scheduler = TDD_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    tddSymb,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
        else:  # FDD Schedulers
            if self.label == "LTE":
                scheduler = LTE_scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.mimoMd,
                    self.layers,
                    dir,
                )
            elif self.schType[0:2] == "PF":
                scheduler = PF_scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            elif self.schType[0:2] == "RR":
                scheduler = RRp_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            elif self.schType[0:2] == "Rr":
                scheduler = RR_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            elif self.schType[0:2] == "SV":
                scheduler = SV_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            elif self.schType[0:2] == "QL":
                scheduler = QL_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            elif self.schType[0:2] == "Q2":
                scheduler = QL2d_Scheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )
            else:  # RR Scheduler by default
                scheduler = IntraSliceScheduler(
                    self.band,
                    self.PRBs,
                    self.dm,
                    self.signLoad,
                    self.ttiBms,
                    self.mimoMd,
                    self.layers,
                    dir,
                    14,
                    self.robustMCS,
                    self.label,
                    self.schType,
                )

        return scheduler

    def setInitialConfig(self):
        """This method sets initial Slice configuration according to service requirements."""
        self.dly2scs(self.reqDelay)
        if (
            self.band == "n257"
            or self.band == "n258"
            or self.band == "n260"
            or self.band == "n261"
        ):
            self.tdd = True
            numReftable = {"60kHz": 1, "120kHz": 2}
            self.numRefFactor = numReftable[self.scs]
            if self.reqDLconnections > 0 and self.reqULconnections == 0:
                self.tddSymb = 14
            if self.reqULconnections > 0 and self.reqDLconnections == 0:
                self.tddSymb = 0
            if self.reqULconnections > 0 and self.reqDLconnections > 0:
                DLfactor = float(self.reqDLconnections * self.reqThroughputDL) / (
                    self.reqDLconnections * self.reqThroughputDL
                    + self.reqULconnections * self.reqThroughputUL
                )
                self.tddSymb = int(14 * DLfactor)
                # print(DLfactor,self.tddSymb)
        else:
            # self.tdd = False
            self.numRefFactor = self.ttiBms

        if self.reqAvailability == "High":
            self.robustMCS = True

        if "mMTC" in self.label:
            self.signLoad = 0.003

    def dly2scs(self, delay):
        """This method sets Slice numerology depending on delay requirements."""
        if (
            self.band == "n257"
            or self.band == "n258"
            or self.band == "n260"
            or self.band == "n261"
        ):  # FR2
            if delay <= 2.5:
                self.scs = "120kHz"
                self.ttiBms = 8
            else:
                self.scs = "60kHz"
                self.ttiBms = 4
        else:  # FR1
            if delay <= 5:
                self.scs = "60kHz"
                self.ttiBms = 4
            elif delay <= 10:
                self.scs = "30kHz"
                self.ttiBms = 2
            else:
                self.scs = "15kHz"
                self.ttiBms = 1

    def updateConfig(self, n):
        """This method updates Slice allocated PRBs."""
        self.PRBs = n
        self.schedulerDL.nrbUEmax = self.PRBs
        if self.label != "LTE":
            self.schedulerUL.nrbUEmax = self.PRBs
        if self.mimoMd == "MU":
            self.schedulerDL.queue.updateSize(self.PRBs * self.layers)
            if self.label != "LTE":
                self.schedulerUL.queue.updateSize(self.PRBs * self.layers)
        else:
            self.schedulerDL.queue.updateSize(self.PRBs)
            if self.label != "LTE":
                self.schedulerUL.queue.updateSize(self.PRBs)
