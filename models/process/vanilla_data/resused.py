# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

C = 5000
packetsfile = 'sliceResUse'
# read the data from the file
with open(packetsfile, 'r') as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
    data = list(reader)
    data = np.array(data)

pcktsvideo = []
pcktsvolte = []
pcktstel = []
pcktsbe = []

for row in data:
    if row[2] == 'Video':
        pcktsvideo.append(row)
    elif row[2] == 'URLLC':
        pcktsvolte.append(row)
    elif row[2] == 'VoLTE':
        pcktstel.append(row)

# First, print the packet arrived to the UE

x = [a[0] for a in pcktsvideo][:240]

x = list(range(0, 5001, 50))
y_pcktsvideo = [a[1] for a in pcktsvideo]
y_pcktsvolte = [a[1] for a in pcktsvolte]
y_pcktstel = [a[1] for a in pcktstel]
y_pcktsbe = [a[1] for a in pcktsbe]

# convert array of strings to array of floats
x = np.array(x, dtype=float)
y_pcktsvideo = np.array(y_pcktsvideo, dtype=float)
y_pcktsvolte = np.array(y_pcktsvolte, dtype=float)
y_pcktstel = np.array(y_pcktstel, dtype=float)

# reshape x to have (0,) shape

# make x and y_pcktsvideo same shape
x = x.reshape(-1, 1)
y_pcktsvideo = y_pcktsvideo.reshape(-1, 1)
y_pcktsvolte = y_pcktsvolte.reshape(-1, 1)
y_pcktstel = y_pcktstel.reshape(-1, 1)



# plots y_pcktstel, y_pcktsvolte, y_pcktsvideo vs x, put labels, legend, etc and make it look nice and smooth the graph
lineWidth = 0.5


plt.plot(x, y_pcktsvolte, label = "eMBB-0", linewidth=lineWidth)
plt.plot(x, y_pcktsvideo, label = "eMBB-1", linewidth=lineWidth)
plt.plot(x, y_pcktstel, label = "eMBB-2", linewidth=lineWidth)
plt.xlabel('Tiempo (ms)')
plt.ylabel('PRBs')

ax = plt.gca()

plt.autoscale()
ax.xaxis.set_major_locator(MultipleLocator(500))

lettersize = 12

plt.legend( prop={"size": lettersize}, fancybox=True)

sns.despine()
#plt.legend(loc='center right')
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.grid(which="major", color="#CCCCCC", linestyle="--")

# increase axis label font size
ax.xaxis.label.set_size(lettersize)
ax.yaxis.label.set_size(lettersize)

# increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=10)

plt.savefig(
        'resUsed' + ".svg"
    )

plt.savefig(
        'resUsed' + ".pdf"
    )
plt.show()




