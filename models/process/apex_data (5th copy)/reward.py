# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 0.0, 0.0, 1.0, 124]['URLLC', 0.0, 0.0, 1.0, 599]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0.0, 0.0, 1.0, 200]['URLLC', 0.0, 0.0, 1.0, 870]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0.5, 0.5, 1.0, 76]['URLLC', 0.0, 0.0, 1.0, 776]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.2, 0.2, 1.0, 768]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0.7, 1.0, 0.7, 8]['URLLC', 0.9, 1.0, 0.9, 723]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 1.0, 1.0, 1.0, 432]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 101]['URLLC', 0.9, 1.0, 0.9, 160]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 13]['URLLC', 0.6, 1.0, 0.6, 12]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.6, 1.0, 0.6, 15]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 108]['URLLC', 0.6, 1.0, 0.6, 11]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 10]['URLLC', 0.6, 1.0, 0.6, 13]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7, 1.0, 0.7, 8]['URLLC', 0.7, 1.0, 0.7, 18]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 105]['URLLC', 0.8, 1.0, 0.8, 33]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 8]['URLLC', 1.0, 1.0, 1.0, 67]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.8, 1.0, 0.8, 106]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 101]['URLLC', 0.7, 1.0, 0.7, 11]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 0.7, 1.0, 0.7, 12]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.8, 1.0, 0.8, 28]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 93]['URLLC', 0.9, 1.0, 0.9, 52]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 11]['URLLC', 0.7, 1.0, 0.7, 16]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.8, 1.0, 0.8, 19]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 82]['URLLC', 0.7200000000000001, 0.8, 0.9, 30]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 12]['URLLC', 1.0, 1.0, 1.0, 54]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7, 1.0, 0.7, 10]['URLLC', 0.9, 1.0, 0.9, 43]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 102]['URLLC', 1.0, 1.0, 1.0, 97]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 11]['URLLC', 0.9, 1.0, 0.9, 86]['VoLTE', 0, 0.0, 1.0, 495]['Video', 0.8, 1.0, 0.8, 12]['URLLC', 0.8, 1.0, 0.8, 13]['VoLTE', 0.0, 0.0, 1.0, 495]['Video', 0.8, 1.0, 0.8, 98]['URLLC', 0.8, 1.0, 0.8, 25]['VoLTE', 0.0, 0.0, 1.0, 459]['Video', 0.9, 1.0, 0.9, 6]['URLLC', 0.7200000000000001, 0.8, 0.9, 52]['VoLTE', 0.2, 0.2, 1.0, 500]['Video', 0.8, 1.0, 0.8, 6]['URLLC', 0.8, 1.0, 0.8, 18]['VoLTE', 0.0, 0.0, 1.0, 461]['Video', 0.8, 1.0, 0.8, 107]['URLLC', 0.8, 1.0, 0.8, 43]['VoLTE', 0.2, 0.2, 1.0, 493]['Video', 1.0, 1.0, 1.0, 16]['URLLC', 1.0, 1.0, 1.0, 53]['VoLTE', 0.0, 0.0, 1.0, 466]['Video', 0.8, 1.0, 0.8, 6]['URLLC', 0.8, 1.0, 0.8, 67]['VoLTE', 0.2, 0.2, 1.0, 402]['Video', 0.8, 1.0, 0.8, 90]['URLLC', 0.7, 1.0, 0.7, 18]['VoLTE', 0.2, 0.2, 1.0, 388]['Video', 0.7200000000000001, 0.8, 0.9, 11]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 1.0, 1.0, 1.0, 388]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.8, 1.0, 0.8, 38]['VoLTE', 1.0, 1.0, 1.0, 244]['Video', 0.8, 1.0, 0.8, 110]['URLLC', 1.0, 1.0, 1.0, 62]['VoLTE', 1.0, 1.0, 1.0, 120]['Video', 0.9, 1.0, 0.9, 4]['URLLC', 0.8, 1.0, 0.8, 72]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 0.8, 1.0, 0.8, 18]['VoLTE', 0.7, 1.0, 0.7, 14]['Video', 0.8, 1.0, 0.8, 105]['URLLC', 0.8, 1.0, 0.8, 16]['VoLTE', 0.8, 1.0, 0.8, 8]['Video', 0.9, 1.0, 0.9, 12]['URLLC', 0.9, 1.0, 0.9, 55]['VoLTE', 0.7, 1.0, 0.7, 13]['Video', 0.8, 1.0, 0.8, 5]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 0.9, 1.0, 0.9, 11]['Video', 0.8, 1.0, 0.8, 109]['URLLC', 0.7, 1.0, 0.7, 13]['VoLTE', 0.9, 1.0, 0.9, 44]['Video', 0.7200000000000001, 0.8, 0.9, 5]['URLLC', 0.8, 1.0, 0.8, 44]['VoLTE', 1.0, 1.0, 1.0, 55]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.8, 0.8, 1.0, 73]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.8, 1.0, 0.8, 110]['URLLC', 0.8, 1.0, 0.8, 98]['VoLTE', 0.9, 1.0, 0.9, 33]['Video', 0.9, 1.0, 0.9, 11]['URLLC', 0.7, 1.0, 0.7, 17]['VoLTE', 1.0, 1.0, 1.0, 65]['Video', 0.8, 1.0, 0.8, 13]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 1.0, 1.0, 1.0, 13]['Video', 0.8, 1.0, 0.8, 94]['URLLC', 0.8, 1.0, 0.8, 36]['VoLTE', 1.0, 1.0, 1.0, 40]['Video', 0.7200000000000001, 0.8, 0.9, 10]['URLLC', 1.0, 1.0, 1.0, 97]['VoLTE', 1.0, 1.0, 1.0, 60]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0.8, 1.0, 0.8, 104]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 0.9, 1.0, 0.9, 95]['URLLC', 0.8, 1.0, 0.8, 17]['VoLTE', 0.9, 1.0, 0.9, 32]['Video', 0.9, 1.0, 0.9, 10]['URLLC', 0.8, 1.0, 0.8, 10]['VoLTE', 1.0, 1.0, 1.0, 31]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.8, 1.0, 0.8, 60]['VoLTE', 1.0, 1.0, 1.0, 51]['Video', 0.8, 1.0, 0.8, 91]['URLLC', 0.7, 1.0, 0.7, 10]['VoLTE', 0.9, 1.0, 0.9, 49]['Video', 0.9, 1.0, 0.9, 6]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 1.0, 1.0, 1.0, 49]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.8, 1.0, 0.8, 50]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 92]['URLLC', 0.8, 1.0, 0.8, 10]['VoLTE', 0.9, 1.0, 0.9, 49]['Video', 0.9, 1.0, 0.9, 8]['URLLC', 0.9, 1.0, 0.9, 11]['VoLTE', 1.0, 1.0, 1.0, 62]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.8, 1.0, 0.8, 44]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.8, 1.0, 0.8, 91]['URLLC', 1.0, 1.0, 1.0, 64]['VoLTE', 0.9, 1.0, 0.9, 32]['Video', 0.7200000000000001, 0.8, 0.9, 11]['URLLC', 0.8, 1.0, 0.8, 110]['VoLTE', 1.0, 1.0, 1.0, 63]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0, 1.0, 0.8, 0]['VoLTE', 0.9, 1.0, 0.9, 9]['Video', 0.9, 1.0, 0.9, 95]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 29]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 49]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 19]['Video', 0.8, 1.0, 0.8, 96]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 17]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 49]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 91]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 43]['Video', 0.9, 1.0, 0.9, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 80]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 20]['Video', 0.8, 1.0, 0.8, 106]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 41]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 78]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 13]['Video', 0.8, 1.0, 0.8, 97]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 44]['Video', 0.9, 1.0, 0.9, 19]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 67]['Video', 0.8, 1.0, 0.8, 14]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 15]['Video', 0.8, 1.0, 0.8, 97]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 32]['Video', 0.7200000000000001, 0.8, 0.9, 21]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 63]"


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







































