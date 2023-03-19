
# Py5cheSim

Py5cheSim is a flexible and open-source simulator based on Python and specially oriented to simulate cell capacity in 3GPP 5G networks and beyond. To the best of our knowledge, Py5cheSim is the first simulator that supports Network Slicing at the Radio Access Network (RAN), one of the main innovations of 5G.

Py5cheSim was designed to simplify new schedulers implementation. The main network model abstractions are contained in the simulator Core, so there is no need to have deep knowledge on that field to run a simulation or to integrate a new scheduler. In this way Py5cheSim provides a framework for 5G new scheduler algorithms implementation in a straightforward and intuitive way.

The tool used to implement Discrete Event Simulation was SimPy and code documentation was made using pydoctor.

Py5cheSim is build on the next modules:

- UE.py: UE parameters and traffic generation.
- Cell.py: Cell configuration and statistics management.
- Slice.py: Slice configuration.
- IntraSliceSch.py: Base intra slice scheduler implementation.
- InterSliceSch.py: Base inter slice scheduler implementation.
- Scheds Intra.py: Other intra slice schedulers implementation.
- Scheds Inter.py: Other inter slice schedulers implementation.
- simulation.py: Is the simulation script. It configures and runs a simulation.
- Results.py: Provides auxiliary methods to present simulation results, and configure traffic profiles.

In addition, tools for developing new schedulers can be found in the lib folder. The following tools are available in the lib folder:

   - UE stats.py: provides user statistics
   - Slice stats.py: provides slice statistics
   - Scheduler Utils.py: offers tools for scheduling
   - AI Slice Scheduler Utils.py: provides tools for designing inter-slice schedulers
   - AI Scheduler Utils.py: offers tools for designing intra-slice schedulers.

Examples of the implementation of these algorithms can be found in Scheds_Inter.py and Scheds_Intra.py.

In addition, two files, apex.py and vanilla.py, can be found. Both are inter-slice scheduling algorithms based on DQN. These files are temporary and will be integrated into Scheds Inter.py.

## Required libraries

In order to install the Py5CheSim\_DRL with the corresponding libraries you first must install Anaconda software from main page [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

Once you have already installed conda you can create an environment with 

```
conda create -n py5chesim_dqn
``` 

Now, the conda environment is created, to activate it just

```
conda activate py5chesim_dqn
```

Following, you should install python 3.8 in orden to be compatible with tensorflow

```
conda install python=3.8
```

Then, the libraries needed in order to run the simulator with the dqn modules included are in requirements.txt

```
pip3 install -r requirements.txt
```

At this point is all ready to run.

## First steps 

All the traffic profiles, scheduling algorithms, and cell parameters are configured in the Simulation.py module, where the entire simulation setup is located. For instance, users can choose the inter-slice scheduler type, bandwidth, simulation time, and inter-slice granularity. The traffic profile is defined by creating a UEgroup object that specifies its characteristics. 

The UEgroup object takes the following attributes:

- number of users in downlink,
- number of users in uplink,
- average packet size in downlink,
- average packet size in uplink,
- average packet arrival time in downlink,
- average packet arrival time in uplink,
- percentage of simulation time during which data exchange is started,
- percentage of simulation time during which data exchange is stopped,
- packet size distribution,
- packet arrival distribution,
- label,
- delay requirement,
- availability,
- intra-slice scheduler type,
- MIMO mode,
- number of layers,
- cell,
- simulation time,
- time between measurements,
- simpy environment,
- SINR initialization parameter

An example configuration could be:

UEgroup1 = UEgroup(2, 0, 300, 0, 1, 0, 0.25, 1, 'Pareto', 'Pareto', "eMBB-0", 20, "", "Rr", "SU", 4, cell1, t_sim, measInterv, env, "S37")

As many UEgroup objects as desired can be created, and each will be associated with a different slice. An example is provided in the main module, and other traffic profile examples can be found in the trafic_profiles folder in the same repository.

Once the desired configuration is set, the simulation can be run from a terminal using the following command:

```
python3 simulation.py
```

After completing the simulation, all statistics will be stored in the Statistics folder, and the figures will be saved in the Figures folder.

## About the project

The simulator is the product of Gabriela Pereyra's master's thesis (Faculty of Engineering, UdelaR, 2021). The current version was modified during the development of Lucas Inglés' thesis (Faculty of Engineering, UdelaR, 2023). Both works were supervised by Claudina Rattaro and Pablo Belzarena.

