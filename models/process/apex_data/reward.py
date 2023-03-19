# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 1, 0.0, 1.0, 0]['URLLC', 0, 0.0, 1.0, 594]['VoLTE', 0, 0.0, 1.0, 139]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 594]['VoLTE', 0.0, 0.0, 0.4, 181]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 555]['VoLTE', 0.18, 0.3, 0.6, 1]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 600]['VoLTE', 0.3, 1.0, 0.3, 3]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 593]['VoLTE', 0.2, 1.0, 0.2, 91]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 524]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.2, 0.2, 1.0, 572]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 538]['VoLTE', 0.2, 1.0, 0.2, 92]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 503]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 498]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 491]['VoLTE', 0.3, 1.0, 0.3, 85]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 488]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 298]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 132]['VoLTE', 0.2, 1.0, 0.2, 99]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 10]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.4, 1.0, 0.4, 7]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 12]['VoLTE', 0.2, 1.0, 0.2, 95]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 19]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 13]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 48]['VoLTE', 0.2, 1.0, 0.2, 93]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 88]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 11]['VoLTE', 0.4, 1.0, 0.4, 1]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 52]['VoLTE', 0.2, 1.0, 0.2, 95]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 14]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 11]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 0, 0.0, 1.0, 236]['URLLC', 0.9, 1.0, 0.9, 66]['VoLTE', 0.2, 1.0, 0.2, 79]['Video', 0.0, 0.0, 1.0, 400]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 0.2, 0.2, 1.0, 395]['URLLC', 0.8, 1.0, 0.8, 13]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 368]['URLLC', 0.9, 1.0, 0.9, 44]['VoLTE', 0.2, 1.0, 0.2, 90]['Video', 1.0, 1.0, 1.0, 230]['URLLC', 1.0, 1.0, 1.0, 96]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 192]['URLLC', 0.9, 1.0, 0.9, 12]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 8]['URLLC', 0.9, 1.0, 0.9, 64]['VoLTE', 0.2, 1.0, 0.2, 92]['Video', 0.9, 1.0, 0.9, 10]['URLLC', 0.9, 1.0, 0.9, 13]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 1.0, 1.0, 1.0, 6]['URLLC', 0.9, 1.0, 0.9, 18]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 17]['URLLC', 0.9, 1.0, 0.9, 31]['VoLTE', 0.2, 1.0, 0.2, 87]['Video', 1.0, 1.0, 1.0, 10]['URLLC', 1.0, 1.0, 1.0, 86]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1.0, 1.0, 1.0, 96]['URLLC', 1.0, 1.0, 1.0, 16]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 65]['URLLC', 0.9, 1.0, 0.9, 47]['VoLTE', 0.2, 1.0, 0.2, 89]['Video', 1.0, 1.0, 1.0, 78]['URLLC', 1.0, 1.0, 1.0, 59]['VoLTE', 0.12, 0.3, 0.4, 4]['Video', 1.0, 1.0, 1.0, 44]['URLLC', 0.9, 1.0, 0.9, 11]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 1.0, 1.0, 1.0, 8]['URLLC', 0.9, 1.0, 0.9, 40]['VoLTE', 0.3, 1.0, 0.3, 90]['Video', 1.0, 1.0, 1.0, 65]['URLLC', 1.0, 1.0, 1.0, 59]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1.0, 1.0, 1.0, 28]['URLLC', 0.9, 1.0, 0.9, 15]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0.9, 1.0, 0.9, 52]['VoLTE', 0.2, 1.0, 0.2, 81]['Video', 1.0, 1.0, 1.0, 19]['URLLC', 0.9, 1.0, 0.9, 16]['VoLTE', 0.12, 0.3, 0.4, 3]['Video', 1.0, 1.0, 1.0, 28]['URLLC', 0.8, 1.0, 0.8, 15]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 1.0, 1.0, 1.0, 93]['URLLC', 0.9, 1.0, 0.9, 45]['VoLTE', 0.2, 1.0, 0.2, 92]['Video', 1.0, 1.0, 1.0, 56]['URLLC', 1.0, 1.0, 1.0, 74]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 76]['URLLC', 0.9, 1.0, 0.9, 15]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 26]['URLLC', 0.9, 1.0, 0.9, 48]['VoLTE', 0.2, 1.0, 0.2, 83]['Video', 1.0, 1.0, 1.0, 84]['URLLC', 1.0, 1.0, 1.0, 75]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 1.0, 1.0, 1.0, 71]['URLLC', 0.9, 1.0, 0.9, 15]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 38]['URLLC', 0.9, 1.0, 0.9, 33]['VoLTE', 0.2, 1.0, 0.2, 73]['Video', 1.0, 1.0, 1.0, 87]['URLLC', 1.0, 1.0, 1.0, 71]['VoLTE', 0.12, 0.3, 0.4, 5]['Video', 1.0, 1.0, 1.0, 76]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0.9, 1.0, 0.9, 54]['VoLTE', 0.2, 1.0, 0.2, 104]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0.8, 1.0, 0.8, 13]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 26]['URLLC', 0.8, 1.0, 0.8, 16]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 100]['URLLC', 0.9, 1.0, 0.9, 36]['VoLTE', 0.2, 1.0, 0.2, 94]['Video', 1.0, 1.0, 1.0, 115]['URLLC', 1.0, 1.0, 1.0, 54]['VoLTE', 0.15, 0.3, 0.5, 6]['Video', 1.0, 1.0, 1.0, 51]['URLLC', 1.0, 1.0, 1.0, 18]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 22]['URLLC', 0.9, 1.0, 0.9, 34]['VoLTE', 0.2, 1.0, 0.2, 86]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0, 1.0, 0.8, 0]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.2, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 42]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 75]['Video', 1.0, 1.0, 1.0, 75]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 4]['Video', 1.0, 1.0, 1.0, 57]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 4]['Video', 1.0, 1.0, 1.0, 28]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 73]['Video', 1.0, 1.0, 1.0, 72]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 3]['Video', 0.9, 1.0, 0.9, 6]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0, 1.0, 0.4, 0]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 99]['Video', 0.8, 1.0, 0.8, 12]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 44]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 77]['Video', 1.0, 1.0, 1.0, 79]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 1]['Video', 1.0, 1.0, 1.0, 32]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 2]['Video', 1.0, 1.0, 1.0, 113]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 74]['Video', 1.0, 1.0, 1.0, 80]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.12, 0.3, 0.4, 4]['Video', 1.0, 1.0, 1.0, 62]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 5]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 92]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.15, 0.3, 0.5, 2]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.4, 1.0, 0.4, 3]['Video', 1.0, 1.0, 1.0, 4]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.2, 1.0, 0.2, 99]['Video', 1.0, 1.0, 1.0, 1]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0, 0.3, 0.7, 0]['Video', 0, 0.2, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0, 0.7, 1.0, 0]['Video', 0, 0.2, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]['Video', 1, 0.0, 1.0, 0]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1, 0.0, 1.0, 0]"


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







































