# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 94]['VoLTE', 0, 0.0, 1.0, 594]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 152]['VoLTE', 0.0, 0.0, 1.0, 594]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 80]['VoLTE', 0.2, 0.2, 1.0, 559]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 12]['VoLTE', 0.2, 0.2, 1.0, 600]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 10]['VoLTE', 0.0, 0.0, 1.0, 593]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 8]['VoLTE', 0.2, 0.2, 1.0, 525]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 72]['VoLTE', 0.2, 0.2, 1.0, 570]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 3]['VoLTE', 1.0, 1.0, 1.0, 527]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 9]['VoLTE', 1.0, 1.0, 1.0, 502]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 50]['VoLTE', 1.0, 1.0, 1.0, 502]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 92]['VoLTE', 1.0, 1.0, 1.0, 481]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 144]['VoLTE', 1.0, 1.0, 1.0, 482]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 53]['VoLTE', 1.0, 1.0, 1.0, 320]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 112]['VoLTE', 1.0, 1.0, 1.0, 153]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 38]['VoLTE', 0.9, 1.0, 0.9, 29]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 99]['VoLTE', 0.8, 1.0, 0.8, 13]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 24]['VoLTE', 0.7, 1.0, 0.7, 17]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 91]['VoLTE', 0.8, 1.0, 0.8, 15]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 16]['VoLTE', 0.8, 1.0, 0.8, 14]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 5]['VoLTE', 0.9, 1.0, 0.9, 58]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 87]['VoLTE', 0.8, 1.0, 0.8, 17]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 12]['VoLTE', 0.8, 1.0, 0.8, 16]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 3]['VoLTE', 0.9, 1.0, 0.9, 48]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 58]['VoLTE', 1.0, 1.0, 1.0, 96]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 113]['VoLTE', 1.0, 1.0, 1.0, 17]['Video', 0, 0.0, 1.0, 462]['URLLC', 0.9, 1.0, 0.9, 148]['VoLTE', 0.9, 1.0, 0.9, 54]['Video', 0.0, 0.0, 1.0, 627]['URLLC', 1.0, 1.0, 1.0, 39]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 0.2, 0.2, 1.0, 650]['URLLC', 0.9, 1.0, 0.9, 125]['VoLTE', 0.8, 1.0, 0.8, 21]['Video', 1.0, 1.0, 1.0, 505]['URLLC', 1.0, 1.0, 1.0, 32]['VoLTE', 0.9, 1.0, 0.9, 31]['Video', 1.0, 1.0, 1.0, 556]['URLLC', 0.9, 1.0, 0.9, 89]['VoLTE', 1.0, 1.0, 1.0, 64]['Video', 1.0, 1.0, 1.0, 331]['URLLC', 1.0, 1.0, 1.0, 14]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 1.0, 1.0, 1.0, 19]['URLLC', 0.9, 1.0, 0.9, 10]['VoLTE', 0.9, 1.0, 0.9, 43]['Video', 1.0, 1.0, 1.0, 83]['URLLC', 0.9, 1.0, 0.9, 68]['VoLTE', 1.0, 1.0, 1.0, 77]['Video', 1.0, 1.0, 1.0, 20]['URLLC', 1.0, 1.0, 1.0, 11]['VoLTE', 1.0, 1.0, 1.0, 18]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 0.9, 1.0, 0.9, 9]['VoLTE', 0.9, 1.0, 0.9, 67]['Video', 1.0, 1.0, 1.0, 7]['URLLC', 0.9, 1.0, 0.9, 78]['VoLTE', 0.9, 1.0, 0.9, 23]['Video', 1.0, 1.0, 1.0, 5]['URLLC', 1.0, 1.0, 1.0, 9]['VoLTE', 0.8, 1.0, 0.8, 11]['Video', 1.0, 1.0, 1.0, 13]['URLLC', 0.8, 1.0, 0.8, 8]['VoLTE', 0.9, 1.0, 0.9, 37]['Video', 1.0, 1.0, 1.0, 139]['URLLC', 0.9, 1.0, 0.9, 66]['VoLTE', 0.9, 1.0, 0.9, 58]['Video', 1.0, 1.0, 1.0, 127]['URLLC', 0.9, 1.0, 0.9, 4]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 1.0, 1.0, 1.0, 62]['URLLC', 0.9, 1.0, 0.9, 8]['VoLTE', 0.9, 1.0, 0.9, 71]['Video', 1.0, 1.0, 1.0, 73]['URLLC', 0.9, 1.0, 0.9, 68]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 1.0, 1.0, 1.0, 76]['URLLC', 0.9, 1.0, 0.9, 14]['VoLTE', 0.8, 1.0, 0.8, 16]['Video', 0.9, 1.0, 0.9, 8]['URLLC', 0.9, 1.0, 0.9, 6]['VoLTE', 0.9, 1.0, 0.9, 57]['Video', 0.7, 1.0, 0.7, 14]['URLLC', 0.9, 1.0, 0.9, 88]['VoLTE', 0.8, 1.0, 0.8, 15]['Video', 1.0, 1.0, 1.0, 18]['URLLC', 1.0, 1.0, 1.0, 15]['VoLTE', 0.9, 1.0, 0.9, 23]['Video', 0.9, 1.0, 0.9, 54]['URLLC', 0.8, 1.0, 0.8, 6]['VoLTE', 0.9, 1.0, 0.9, 45]['Video', 0.9, 1.0, 0.9, 7]['URLLC', 0.9, 1.0, 0.9, 60]['VoLTE', 1.0, 1.0, 1.0, 54]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 1.0, 1.0, 1.0, 125]['VoLTE', 0.9, 1.0, 0.9, 16]['Video', 0.9, 1.0, 0.9, 27]['URLLC', 0.9, 1.0, 0.9, 201]['VoLTE', 0.9, 1.0, 0.9, 18]['Video', 1.0, 1.0, 1.0, 11]['URLLC', 1.0, 1.0, 1.0, 109]['VoLTE', 1.0, 1.0, 1.0, 56]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.8, 1.0, 0.8, 26]['VoLTE', 0.9, 1.0, 0.9, 17]['Video', 0.6, 1.0, 0.7, 7]['URLLC', 0.8, 1.0, 0.8, 11]['VoLTE', 0.9, 1.0, 0.9, 52]['Video', 1.0, 1.0, 1.0, 51]['URLLC', 0.9, 1.0, 0.9, 10]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 1.0, 1.0, 1.0, 22]['URLLC', 0.9, 1.0, 0.9, 71]['VoLTE', 0.9, 1.0, 0.9, 26]['Video', 1.0, 1.0, 1.0, 19]['URLLC', 0.9, 1.0, 0.9, 11]['VoLTE', 0.9, 1.0, 0.9, 63]['Video', 1.0, 1.0, 1.0, 148]['URLLC', 0.9, 1.0, 0.9, 8]['VoLTE', 0.9, 1.0, 0.9, 19]['Video', 1.0, 1.0, 1.0, 31]['URLLC', 0.9, 1.0, 0.9, 47]['VoLTE', 0.9, 1.0, 0.9, 18]['Video', 1.0, 1.0, 1.0, 79]['URLLC', 1.0, 1.0, 1.0, 108]['VoLTE', 0.9, 1.0, 0.9, 52]['Video', 1.0, 1.0, 1.0, 63]['URLLC', 0.9, 1.0, 0.9, 166]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 1.0, 1.0, 1.0, 8]['URLLC', 1.0, 1.0, 1.0, 59]['VoLTE', 0.8, 1.0, 0.8, 15]['Video', 1.0, 1.0, 1.0, 69]['URLLC', 0.8, 1.0, 0.8, 9]['VoLTE', 0.9, 1.0, 0.9, 52]['Video', 1.0, 1.0, 1.0, 14]['URLLC', 0, 1.0, 0.6, 0]['VoLTE', 0.9, 1.0, 0.9, 16]['Video', 0.9, 1.0, 0.9, 16]['URLLC', 0, 0.2, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 15]['Video', 0.9, 1.0, 0.9, 25]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 69]['Video', 1.0, 1.0, 1.0, 70]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 1.0, 1.0, 1.0, 144]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 16]['Video', 0.8, 1.0, 0.8, 17]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 70]['Video', 0.7, 1.0, 0.7, 7]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 13]['Video', 0.9, 1.0, 0.9, 6]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 12]['Video', 0.8, 1.0, 0.7, 13]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 31]['Video', 1.0, 1.0, 1.0, 13]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 64]['Video', 1.0, 1.0, 1.0, 60]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 10]['Video', 1.0, 1.0, 1.0, 42]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 35]['Video', 1.0, 1.0, 1.0, 86]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 73]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.7, 1.0, 0.7, 8]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 45]['Video', 1.0, 1.0, 1.0, 7]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 83]['Video', 1, 1.0, 1.0, 24]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 17]['Video', 1.0, 1.0, 1.0, 37]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 49]"


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







































