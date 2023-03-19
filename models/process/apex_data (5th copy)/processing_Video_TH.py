# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

C = 5000
packetsfile = 'everyThroughputVideo'
# read the data from the file
with open(packetsfile, 'r') as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace = True)
    data = list(reader)
    data = np.array(data)

ue1 = []
ue2 = []
ue3 = []
ue4 = []
ue5 = []
ue6 = []
ue7 = []
ue8 = []

for row in data:
    if row[2] == 'ue1':
        ue1.append(row)
    elif row[2] == 'ue2':
        ue2.append(row)
    elif row[2] == 'ue3':
        ue3.append(row)
    elif row[2] == 'ue4':
        ue4.append(row)
    elif row[2] == 'ue5':
        ue5.append(row)
    elif row[2] == 'ue6':
        ue6.append(row)
    elif row[2] == 'ue7':
        ue7.append(row)
    elif row[2] == 'ue8':
        ue8.append(row)



# First, print the packet arrived to the UE

x = [a[0] for a in ue1][:51]
y_ue1 = [a[1] for a in ue1][:51]
y_ue2 = [a[1] for a in ue2][:51]
y_ue3 = [a[1] for a in ue3][:51]
y_ue4 = [a[1] for a in ue4][:51]
y_ue5 = [a[1] for a in ue5][:51]
y_ue6 = [a[1] for a in ue6][:51]
y_ue7 = [a[1] for a in ue7][:51]
y_ue8 = [a[1] for a in ue8][:51]





# convert array of strings to array of floats
x = np.array(x, dtype=float)
y_ue1 = np.array(y_ue1, dtype=float)
y_ue2 = np.array(y_ue2, dtype=float)
y_ue3 = np.array(y_ue3, dtype=float)
y_ue4 = np.array(y_ue4, dtype=float)
y_ue5 = np.array(y_ue5, dtype=float)
y_ue6 = np.array(y_ue6, dtype=float)
y_ue7 = np.array(y_ue7, dtype=float)
y_ue8 = np.array(y_ue8, dtype=float)

print(x)
print('\n', y_ue1)
# plots y_pcktstel, y_pcktsvolte, y_pcktsvideo vs x, put labels, legend, etc and make it look nice and smooth the graph
lineWidth = 0.5

plt.plot(x, y_ue1, label = "ue1", linewidth=lineWidth)
plt.plot(x, y_ue2, label = "ue2", linewidth=lineWidth)
plt.plot(x, y_ue3, label = "ue3", linewidth=lineWidth)
plt.plot(x, y_ue4, label = "ue4", linewidth=lineWidth)

# make horizontal line for indicating threshold with a label

#plt.axhline(y=5, color='r', linestyle='--', linewidth=1, label = 'Umbral')





plt.xlabel('Tiempo (ms)')
plt.ylabel('Throughput (Mbps)')
ax = plt.gca()
# Traza una línea horizontal en la posición y = 5
umbral = 4.5
ax.axhline(y=umbral, color='red', linestyle='--')

# Agrega un indicador de umbral en la posición (0, 5.1)
ax.annotate('SLA', xy=(0, umbral + 0.1), color='red')


plt.autoscale()
ax.xaxis.set_major_locator(MultipleLocator(500))

lettersize = 12

plt.legend( prop={"size": lettersize},fancybox=True)

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
        'throughputVideo' + ".svg"
    )

plt.savefig(
        'throughputVideo' + ".pdf"
    )
plt.show()

# save plot as a pdf

