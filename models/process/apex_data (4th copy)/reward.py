# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 0.0, 0.0, 1.0, 139]['URLLC', 0.0, 0.0, 1.0, 604]['VoLTE', 0, 0.0, 1.0, 0]['Video', 0.0, 0.0, 1.0, 184]['URLLC', 0.0, 0.0, 1.0, 873]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.5, 0.5, 1.0, 77]['URLLC', 0.0, 0.0, 1.0, 789]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.2, 0.2, 1.0, 785]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7, 1.0, 0.7, 8]['URLLC', 1.0, 1.0, 1.0, 739]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 1.0, 1.0, 1.0, 445]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 102]['URLLC', 0.9, 1.0, 0.9, 208]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 16]['URLLC', 0.7, 1.0, 0.7, 9]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 6]['URLLC', 0.6, 1.0, 0.6, 16]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 104]['URLLC', 0.6, 1.0, 0.6, 12]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 8]['URLLC', 0.6, 1.0, 0.6, 18]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0.7, 1.0, 0.7, 11]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 85]['URLLC', 0.8, 1.0, 0.8, 39]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.6400000000000001, 0.8, 0.8, 7]['URLLC', 1.0, 1.0, 1.0, 48]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0.9, 1.0, 0.9, 364]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 89]['URLLC', 0.2, 0.2, 1.0, 403]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 12]['URLLC', 0.8, 1.0, 0.8, 461]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.9, 1.0, 0.9, 173]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 89]['URLLC', 0.7, 1.0, 0.7, 19]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.9, 1.0, 0.9, 8]['URLLC', 0.6, 1.0, 0.6, 14]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 108]['URLLC', 0.7, 1.0, 0.7, 20]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 10]['URLLC', 0.8, 1.0, 0.8, 44]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 1.0, 1.0, 1.0, 78]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.8, 1.0, 0.8, 90]['URLLC', 0.8, 1.0, 0.8, 104]['VoLTE', 1, 0.0, 1.0, 0]['Video', 0.7200000000000001, 0.8, 0.9, 8]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0, 0.0, 1.0, 495]['Video', 0.7, 1.0, 0.7, 11]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.0, 0.0, 1.0, 495]['Video', 0.8, 1.0, 0.8, 90]['URLLC', 0.8, 1.0, 0.8, 61]['VoLTE', 0.0, 0.0, 1.0, 460]['Video', 0.9, 1.0, 0.9, 7]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.2, 0.2, 1.0, 500]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.7, 1.0, 0.7, 12]['VoLTE', 0.0, 0.0, 1.0, 468]['Video', 0.8, 1.0, 0.8, 93]['URLLC', 0.8, 1.0, 0.8, 32]['VoLTE', 0.2, 0.2, 1.0, 497]['Video', 0.45, 0.5, 0.9, 7]['URLLC', 0.8, 0.8, 1.0, 55]['VoLTE', 0.0, 0.0, 1.0, 460]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.8, 1.0, 0.8, 79]['VoLTE', 0.2, 0.2, 1.0, 402]['Video', 0.8, 1.0, 0.8, 89]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.2, 0.2, 1.0, 385]['Video', 0.7200000000000001, 0.8, 0.9, 9]['URLLC', 0.7, 1.0, 0.7, 17]['VoLTE', 1.0, 1.0, 1.0, 388]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.8, 1.0, 0.8, 49]['VoLTE', 1.0, 1.0, 1.0, 243]['Video', 0.8, 1.0, 0.8, 83]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 1.0, 1.0, 1.0, 104]['Video', 0.6400000000000001, 0.8, 0.8, 8]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 0.9, 1.0, 0.9, 21]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.8, 1.0, 0.8, 24]['VoLTE', 0.7, 1.0, 0.7, 12]['Video', 0.8, 1.0, 0.8, 108]['URLLC', 0.9, 1.0, 0.9, 65]['VoLTE', 0.7, 1.0, 0.7, 11]['Video', 0.7200000000000001, 0.8, 0.9, 32]['URLLC', 0.7, 1.0, 0.7, 17]['VoLTE', 0.8, 1.0, 0.8, 12]['Video', 0.8, 1.0, 0.8, 108]['URLLC', 0.8, 1.0, 0.8, 49]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.9, 1.0, 0.9, 34]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 0.9, 1.0, 0.9, 45]['Video', 0.8, 1.0, 0.8, 130]['URLLC', 0.7, 1.0, 0.7, 10]['VoLTE', 1.0, 1.0, 1.0, 55]['Video', 0.2, 0.2, 1.0, 23]['URLLC', 0.8, 1.0, 0.8, 34]['VoLTE', 0.9, 1.0, 0.9, 9]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 1.0, 1.0, 1.0, 53]['VoLTE', 0.9, 1.0, 0.9, 21]['Video', 0.8, 1.0, 0.8, 67]['URLLC', 0.6400000000000001, 0.8, 0.8, 67]['VoLTE', 1.0, 1.0, 1.0, 56]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0.7, 1.0, 0.7, 18]['VoLTE', 1.0, 1.0, 1.0, 10]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.7, 1.0, 0.7, 13]['VoLTE', 0.9, 1.0, 0.9, 50]['Video', 0.8, 1.0, 0.8, 100]['URLLC', 0.8, 1.0, 0.8, 50]['VoLTE', 0.9, 1.0, 0.9, 21]['Video', 0.9, 1.0, 0.9, 11]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.8, 1.0, 0.8, 16]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.7, 1.0, 0.7, 21]['VoLTE', 0.9, 1.0, 0.9, 54]['Video', 0.8, 1.0, 0.8, 84]['URLLC', 0.8, 1.0, 0.8, 58]['VoLTE', 0.9, 1.0, 0.9, 21]['Video', 0.7200000000000001, 0.8, 0.9, 8]['URLLC', 0.7, 1.0, 0.7, 18]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0.7, 1.0, 0.7, 15]['VoLTE', 0.9, 1.0, 0.9, 34]['Video', 0.8, 1.0, 0.8, 92]['URLLC', 0.8, 1.0, 0.8, 43]['VoLTE', 1.0, 1.0, 1.0, 63]['Video', 0.45, 0.5, 0.9, 11]['URLLC', 1.0, 1.0, 1.0, 74]['VoLTE', 1.0, 1.0, 1.0, 18]['Video', 0.8, 1.0, 0.8, 5]['URLLC', 0.8, 1.0, 0.8, 93]['VoLTE', 1.0, 1.0, 1.0, 52]['Video', 0.8, 1.0, 0.8, 109]['URLLC', 0.7, 1.0, 0.7, 14]['VoLTE', 0.9, 1.0, 0.9, 17]['Video', 0.45, 0.5, 0.9, 3]['URLLC', 0.7, 1.0, 0.7, 12]['VoLTE', 0.8, 1.0, 0.8, 13]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.6400000000000001, 0.8, 0.8, 46]['VoLTE', 0.9, 1.0, 0.9, 37]['Video', 0.8, 1.0, 0.8, 110]['URLLC', 1.0, 1.0, 1.0, 75]['VoLTE', 1.0, 1.0, 1.0, 56]['Video', 0.45, 0.5, 0.9, 7]['URLLC', 0, 1.0, 0.7, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 8]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 42]['Video', 0.9, 1.0, 0.9, 117]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 64]['Video', 0.7200000000000001, 0.8, 0.9, 34]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 119]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 39]['Video', 0.45, 0.5, 0.9, 30]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 55]['Video', 0.8, 1.0, 0.8, 95]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 10]['Video', 0.7200000000000001, 0.8, 0.9, 7]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 42]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 52]['Video', 0.8, 1.0, 0.8, 104]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 0.9, 1.0, 0.9, 13]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 18]['Video', 0.8, 1.0, 0.8, 18]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 83]['Video', 0.9, 1.0, 0.9, 105]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.8, 0.8, 1.0, 14]['Video', 0.9, 1.0, 0.9, 6]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 64]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 0.8, 1.0, 0.8, 94]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.9, 1.0, 0.9, 8]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 41]['Video', 0.8, 1.0, 0.8, 17]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 68]"


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







































