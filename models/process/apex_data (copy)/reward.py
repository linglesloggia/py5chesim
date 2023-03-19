# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 1, 0.0, 1.0, 0]['URLLC', 0, 0.0, 1.0, 594]['VoLTE', 0, 0.0, 1.0, 147]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 594]['VoLTE', 0.0, 0.0, 0.4, 201]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 561]['VoLTE', 0.21, 0.3, 0.7, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 600]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 592]['VoLTE', 0.2, 1.0, 0.2, 95]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 533]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 570]['VoLTE', 0.4, 1.0, 0.4, 1]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 543]['VoLTE', 0.3, 1.0, 0.3, 98]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 503]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 496]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 497]['VoLTE', 0.2, 1.0, 0.2, 96]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 488]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 311]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 141]['VoLTE', 0.2, 1.0, 0.2, 78]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 24]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 16]['VoLTE', 0.4, 1.0, 0.4, 1]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 12]['VoLTE', 0.2, 1.0, 0.2, 84]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 10]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 19]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 61]['VoLTE', 0.2, 1.0, 0.2, 78]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 12]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 17]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 37]['VoLTE', 0.2, 1.0, 0.2, 91]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 66]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 17]['VoLTE', 0.5, 1.0, 0.5, 3]['Video', 0, 0.0, 1.0, 238]['URLLC', 1.0, 1.0, 1.0, 68]['VoLTE', 0.2, 1.0, 0.2, 89]['Video', 0.0, 0.0, 1.0, 280]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 0.2, 0.2, 1.0, 273]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 221]['URLLC', 0.9, 1.0, 0.9, 37]['VoLTE', 0.2, 1.0, 0.2, 79]['Video', 1.0, 1.0, 1.0, 216]['URLLC', 1.0, 1.0, 1.0, 90]['VoLTE', 0.12, 0.3, 0.4, 4]['Video', 1.0, 1.0, 1.0, 135]['URLLC', 1.0, 1.0, 1.0, 16]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 84]['URLLC', 0.9, 1.0, 0.9, 55]['VoLTE', 0.2, 1.0, 0.2, 87]['Video', 1.0, 1.0, 1.0, 31]['URLLC', 0.9, 1.0, 0.9, 18]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 96]['URLLC', 0.9, 1.0, 0.9, 18]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 53]['URLLC', 0.9, 1.0, 0.9, 64]['VoLTE', 0.2, 1.0, 0.2, 96]['Video', 1.0, 1.0, 1.0, 19]['URLLC', 0.9, 1.0, 0.9, 19]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 21]['URLLC', 0.9, 1.0, 0.9, 10]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0.9, 1.0, 0.9, 69]['VoLTE', 0.2, 1.0, 0.2, 87]['Video', 1.0, 1.0, 1.0, 70]['URLLC', 0.9, 1.0, 0.9, 17]['VoLTE', 0.15, 0.3, 0.5, 5]['Video', 1.0, 1.0, 1.0, 31]['URLLC', 0.8, 1.0, 0.8, 12]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0.9, 1.0, 0.9, 60]['VoLTE', 0.3, 1.0, 0.3, 69]['Video', 1.0, 1.0, 1.0, 93]['URLLC', 0.9, 1.0, 0.9, 12]['VoLTE', 0, 0.3, 1.0, 197]['Video', 1.0, 1.0, 1.0, 41]['URLLC', 0.8, 1.0, 0.8, 15]['VoLTE', 0.0, 0.0, 0.4, 238]['Video', 1.0, 1.0, 1.0, 68]['URLLC', 0.9, 1.0, 0.9, 60]['VoLTE', 0.21, 0.3, 0.7, 24]['Video', 1.0, 1.0, 1.0, 53]['URLLC', 0.9, 1.0, 0.9, 16]['VoLTE', 0.2, 1.0, 0.2, 6]['Video', 1.0, 1.0, 1.0, 52]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.2, 1.0, 0.2, 2]['Video', 1.0, 1.0, 1.0, 30]['URLLC', 0.9, 1.0, 0.9, 41]['VoLTE', 0.2, 1.0, 0.2, 2]['Video', 1.0, 1.0, 1.0, 8]['URLLC', 1.0, 1.0, 1.0, 82]['VoLTE', 0.2, 1.0, 0.2, 2]['Video', 1.0, 1.0, 1.0, 46]['URLLC', 1.0, 1.0, 1.0, 16]['VoLTE', 0.3, 1.0, 0.3, 4]['Video', 1.0, 1.0, 1.0, 42]['URLLC', 0.9, 1.0, 0.9, 35]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 1.0, 1.0, 1.0, 58]['VoLTE', 0.2, 1.0, 0.2, 92]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0.9, 1.0, 0.9, 16]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 0.7, 1.0, 0.7, 4]['URLLC', 0.9, 1.0, 0.9, 36]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 31]['URLLC', 1.0, 1.0, 1.0, 58]['VoLTE', 0.3, 1.0, 0.3, 84]['Video', 1.0, 1.0, 1.0, 60]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 11]['URLLC', 0.9, 1.0, 0.9, 58]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 1.0, 1.0, 1.0, 24]['URLLC', 0.9, 1.0, 0.9, 15]['VoLTE', 0.2, 1.0, 0.2, 89]['Video', 1.0, 1.0, 1.0, 88]['URLLC', 0.8, 1.0, 0.8, 19]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 66]['URLLC', 0.9, 1.0, 0.9, 67]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 50]['URLLC', 0.9, 1.0, 0.9, 9]['VoLTE', 0.3, 1.0, 0.3, 73]['Video', 1.0, 1.0, 1.0, 21]['URLLC', 0.9, 1.0, 0.9, 17]['VoLTE', 0.12, 0.3, 0.4, 2]['Video', 1.0, 1.0, 1.0, 14]['URLLC', 0.9, 1.0, 0.9, 44]['VoLTE', 0, 1.0, 0.4, 0]['Video', 1.0, 1.0, 1.0, 29]['URLLC', 0, 1.0, 0.8, 0]['VoLTE', 0.2, 1.0, 0.2, 91]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0, 0.2, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 0.8, 1.0, 0.8, 6]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 1]['Video', 1.0, 1.0, 1.0, 11]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 87]['Video', 1.0, 1.0, 1.0, 39]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 38]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 115]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.3, 1.0, 0.3, 90]['Video', 1.0, 1.0, 1.0, 96]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 132]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 52]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.3, 1.0, 0.3, 90]['Video', 1.0, 1.0, 1.0, 43]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 36]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 35]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 88]['Video', 1.0, 1.0, 1.0, 83]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 120]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 7]['Video', 1.0, 1.0, 1.0, 53]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 94]['Video', 1.0, 1.0, 1.0, 65]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.12, 0.3, 0.4, 3]['Video', 1.0, 1.0, 1.0, 37]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 25]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 94]['Video', 1.0, 1.0, 1.0, 47]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1.0, 1.0, 1.0, 10]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 0, 1.0, 0.7, 0]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0, 1.0, 0.4, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]"


sublists = eval("[" + string.replace("][", "], [") + "]")


rewardvideo = []
rewardurllc = []
rewardtel = []

for row in sublists:
    if row[0] == 'Video':
        rewardvideo.append(row[1])
    elif row[0] == 'URLLC':
        rewardurllc.append(row[1])
    elif row[0] == 'VoLTE':
        rewardtel.append(row[1])
print('len2, ', len(rewardtel))

x=[]
for i in range(60, 6000, 60):
     x.append(i)

rewardvideo = rewardvideo[:80]
rewardurllc = rewardurllc[:80]
rewardtel = rewardtel[:80]
x = x[:80]


# plots y_pcktstel, y_pcktsvolte, y_pcktsvideo vs x, put labels, legend, etc and make it look nice and smooth the graph
lineWidth = 0.5


plt.plot(x, rewardurllc, label = "eMBB-0", linewidth=lineWidth)
plt.plot(x, rewardvideo, label = "eMBB-1", linewidth=lineWidth)
plt.plot(x, rewardtel, label = "eMBB-2", linewidth=lineWidth)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Reward')

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
        'reward' + ".svg"
    )

plt.savefig(
        'reward' + ".pdf"
    )
plt.show()







































