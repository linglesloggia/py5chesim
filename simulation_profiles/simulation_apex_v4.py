"""This is the simulation script.
Simulation, cell and traffic profile parameters can be set here.
"""
import os
import sys

import simpy

from Cell import *
from Results import *
from UE import *

# ------------------------------------------------------------------------------------------------------
#              Cell & Simulation parameters
# ------------------------------------------------------------------------------------------------------

bw = [
    10
]  # MHz (FR1: 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 90, 100; FR2: 50, 100, 200, 400)
"""List containing each CC's bandwidth for the simulation. """
fr = "FR1"  # FR1 or FR2
"""String with frequency range (FR) to use. 'FR1' for FR1, or 'FR2' for FR2."""
band = "B1"
"""String with used band for simulation. In TDD mode it is important to set correctly a band from the next list: n257, n258, n260, n261."""
tdd = False
"""Boolean indicating if the cell operates in TDD mode."""
buf = 81920  # 10240 #
"""Integer with the maximum Bytes the UE Bearer buffer can tolerate before dropping packets."""
schedulerInter = "AX"  # RRp for improved Round Robin, PFXX for Proportional Fair
"""String indicating the Inter Slice Scheduler to use. For only one Slice simulations use ''.
If the simulation includes more than one slice, set '' for Round Robin, 'RRp' for Round Robin Plus,
or 'PFXY' for Proportional Fair with X=numExp and Y=denExp."""
#                   Simulation parameters
t_sim = 6_000#12_000_000  # 60000 # (ms)
"""Simulation duration in milliseconds."""
debMode = False  # to show queues information by TTI during simulation
"""Boolean indicating if debugging mode is active. In that case, an html log file will be generated with schedulers operation.
Note that in simulations with a high number of UEs this file can turn quite heavy."""
measInterv = 100  # interval between meassures default 1000
"""Time interval (in milliseconds) between meassures for statistics reports."""
interSliceSchGr = 60  # 3000.0 # interSlice scheduler time granularity
"""Inter slice scheduler time granularity in milliseconds."""

# -----------------------------------------------------------------
#              Simulation process activation
# -----------------------------------------------------------------

env = simpy.Environment()
"""Environment instance needed by simpy for runing PEM methods"""
cell1 = Cell("c1", bw, fr, debMode, buf, tdd, interSliceSchGr, schedulerInter)
"""Cell instance for running the simulation"""
interSliceSche1 = cell1.interSliceSched
"""interSliceScheduler instance"""
# embb-1

UEgroup1 = UEgroup(
    4,
    0,
    410,
    0,
    1.2,
    0,
    0,
    1,
    'Pareto',
    'Uniform',
    "Video",
    20,
    "",
    "Rr",
    "SU",
    4,
    cell1,
    t_sim,
    measInterv,
    env,
    "S37",
)  # pareto u   0.64

UEgroup2 = UEgroup(
    4,
    0,
    300,
    0,
    0.6,
    0,
    0,
    0.75,
    'Constant',
    'Uniform',
    "URLLC",
    20,
    "",
    "Rr",
    "SU",
    4,
    cell1,
    t_sim,
    measInterv,
    env,
    "S37",
)  # br vid    2.06

UEgroup3 = UEgroup(
    5,
    0,
    800,
    0,
    1,
    0,
    0.25,
    1,
    'TruncatedNormal',
    'Uniform',
    "VoLTE",
    20,
    "",
    "Rr",
    "SU",
    4,
    cell1,
    t_sim,
    measInterv,
    env,
    "S37",
)  # br vid    2.06

UEgroup4 = UEgroup(
    0,
    0,
    1,
    0,
    5,
    0,
    0,
    1,    
    'Pareto',
    'Pareto',
    "BestEffort",
    20,
    "",
    "Rr",
    "SU",
    4,
    cell1,
    t_sim,
    measInterv,
    env,
    "S37",
)




"""Group of users with defined traffic profile, capabilities and service requirements for which the sumulation will run.

More than one can be instantiated in one simulation.
For each one of them, the UEgroup instance must be added in the UEgroups list.

UEgroupN = UEgroup(nuDL,nuUL,pszDL,pszUL,parrDL,parrUL, tstart, tend, distP, distA,
label,dly,avlty,schedulerType,mimo_mode,layers,cell,hdr,t_sim,measInterv,env,sinr):

label: must contain substring according to the type of service: eMBB, mMTC, URLLC\n
schedulerType: RR: Rounf Robin, PF: Proportional Fair (10, 11)\n
mimo_mode: SU, MU\n
layers: in SU-MIMO is the number of layers/UE, in MU-MIMO is the number of simultaneous UE to serve with the same resources\n
sinr: is a string starting starting with S if all ues have the same sinr or D if not. The value next will be the initial sinr of each ue or the maximum."""

# UEgroup2 = UEgroup(3,3,800000,300,1,10,'eMBB-1',10,'','RR','',1,cell1,t_sim,measInterv,env,'D37')

# Set UEgroups list according to the defined groups!!!
UEgroups = [UEgroup1, UEgroup2 ,UEgroup3, UEgroup4]
"""UE group list for the configured simulation"""
#           Slices creation
for ueG in UEgroups:
    interSliceSche1.createSlice(
        ueG.req["reqDelay"],
        ueG.req["reqThroughputDL"],
        ueG.req["reqThroughputUL"],
        ueG.req["reqAvailability"],
        ueG.num_usersDL,
        ueG.num_usersUL,
        band,
        debMode,
        ueG.mmMd,
        ueG.lyrs,
        ueG.label,
        ueG.sch,
    )

#      Schedulers activation (inter/intra)

procCell = env.process(cell1.updateStsts(env, interv=measInterv, tSim=t_sim))
procInter = env.process(interSliceSche1.resAlloc(env))
for ueG in UEgroups:
    ueG.activateSliceScheds(interSliceSche1, env)

# ----------------------------------------------------------------
env.run(until=t_sim)
# ----------------------------------------------------------------

#      Closing statistic and debugging files

for slice in list(cell1.slicesStsts.keys()):
    cell1.slicesStsts[slice]["DL"].close()
    cell1.slicesStsts[slice]["UL"].close()
for slice in list(interSliceSche1.slices.keys()):
    interSliceSche1.slices[slice].schedulerDL.dbFile.close()
    if slice != "LTE":
        interSliceSche1.slices[slice].schedulerUL.dbFile.close()

# ----------------------------------------------------------------
#                          RESULTS
# ----------------------------------------------------------------
# Show average PLR and Throughput in any case simulation and plots
for UEg in UEgroups:
    print(
        Format.CBOLD
        + Format.CBLUE
        + "\n--------------------------------------------------"
        + Format.CEND
    )
    print(
        Format.CBOLD
        + Format.CBLUE
        + "                 SLICE: "
        + UEg.label
        + "                  "
        + Format.CEND
    )
    print(
        Format.CBOLD
        + Format.CBLUE
        + "--------------------------------------------------\n"
        + Format.CEND
    )
    UEg.printSliceResults(interSliceSche1, t_sim, bw, measInterv)
print(
    Format.CBOLD
    + Format.CBLUE
    + "\n--------------------------------------------------"
    + Format.CEND
)
