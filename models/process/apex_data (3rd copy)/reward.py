# open a file and read the data, only keep the rows that has 'eMBB-0' in the second column

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

string = "['Video', 1, 0.0, 1.0, 0]['URLLC', 0, 0.0, 1.0, 214]['VoLTE', 0, 0.0, 1.0, 577]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.0, 0.0, 1.0, 310]['VoLTE', 0.0, 0.0, 1.0, 654]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.3, 0.3, 1.0, 280]['VoLTE', 0.2, 0.2, 1.0, 570]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 216]['VoLTE', 0.2, 0.2, 1.0, 581]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 142]['VoLTE', 1.0, 1.0, 1.0, 527]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 4]['VoLTE', 1.0, 1.0, 1.0, 539]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.5, 1.0, 0.5, 2]['VoLTE', 1.0, 1.0, 1.0, 523]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.6, 1.0, 0.6, 3]['VoLTE', 0.9, 1.0, 0.9, 524]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 5]['VoLTE', 1.0, 1.0, 1.0, 392]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.8, 1.0, 0.8, 8]['VoLTE', 0.9, 1.0, 0.9, 255]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 45]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 94]['VoLTE', 0.6, 1.0, 0.6, 13]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 101]['VoLTE', 0.7, 1.0, 0.7, 12]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 26]['VoLTE', 0.7, 1.0, 0.7, 11]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 74]['VoLTE', 0.8, 1.0, 0.8, 13]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 24]['VoLTE', 0.8, 1.0, 0.8, 16]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 69]['VoLTE', 0.9, 1.0, 0.9, 96]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 23]['VoLTE', 1.0, 1.0, 1.0, 9]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.9, 1.0, 0.9, 61]['VoLTE', 0.8, 1.0, 0.8, 10]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 145]['VoLTE', 0.9, 1.0, 0.9, 84]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 153]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 1, 0.0, 1.0, 0]['URLLC', 1.0, 1.0, 1.0, 96]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 7]['VoLTE', 0.9, 1.0, 0.9, 100]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.6, 1.0, 0.6, 9]['VoLTE', 1.0, 1.0, 1.0, 7]['Video', 1, 0.0, 1.0, 0]['URLLC', 0.7, 1.0, 0.7, 3]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0, 0.0, 1.0, 457]['URLLC', 0.8, 1.0, 0.8, 4]['VoLTE', 0.9, 1.0, 0.9, 87]['Video', 0.0, 0.0, 1.0, 643]['URLLC', 0.9, 1.0, 0.9, 37]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.2, 0.2, 1.0, 700]['URLLC', 1.0, 1.0, 1.0, 101]['VoLTE', 0.9, 1.0, 0.9, 8]['Video', 1.0, 1.0, 1.0, 602]['URLLC', 1.0, 1.0, 1.0, 121]['VoLTE', 0.9, 1.0, 0.9, 85]['Video', 1.0, 1.0, 1.0, 495]['URLLC', 1.0, 1.0, 1.0, 63]['VoLTE', 0.9, 1.0, 0.9, 9]['Video', 0.9, 1.0, 0.9, 402]['URLLC', 0.7, 1.0, 0.7, 4]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 1.0, 1.0, 1.0, 104]['URLLC', 0.6, 1.0, 0.6, 5]['VoLTE', 0.9, 1.0, 0.9, 76]['Video', 0.6, 1.0, 0.6, 10]['URLLC', 0.7, 1.0, 0.7, 5]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 0.5, 1.0, 0.5, 8]['URLLC', 0.8, 1.0, 0.8, 6]['VoLTE', 0.9, 1.0, 0.9, 17]['Video', 0.6, 1.0, 0.6, 6]['URLLC', 0.9, 1.0, 0.9, 41]['VoLTE', 0.9, 1.0, 0.9, 100]['Video', 0.7, 1.0, 0.7, 8]['URLLC', 1.0, 1.0, 1.0, 120]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.7, 1.0, 0.7, 9]['URLLC', 1.0, 1.0, 1.0, 125]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 0.8, 1.0, 0.8, 6]['URLLC', 0.9, 1.0, 0.9, 41]['VoLTE', 0.9, 1.0, 0.9, 74]['Video', 1.0, 1.0, 1.0, 11]['URLLC', 1.0, 1.0, 1.0, 88]['VoLTE', 0.9, 1.0, 0.9, 17]['Video', 0.8, 1.0, 0.8, 124]['URLLC', 0.9, 1.0, 0.9, 23]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 1.0, 1.0, 1.0, 38]['URLLC', 0.9, 1.0, 0.9, 54]['VoLTE', 0.9, 1.0, 0.9, 105]['Video', 0.9, 1.0, 0.9, 179]['URLLC', 1.0, 1.0, 1.0, 123]['VoLTE', 1.0, 1.0, 1.0, 18]['Video', 1.0, 1.0, 1.0, 105]['URLLC', 1.0, 1.0, 1.0, 134]['VoLTE', 0.9, 1.0, 0.9, 18]['Video', 0.8, 1.0, 0.8, 27]['URLLC', 0.9, 1.0, 0.9, 66]['VoLTE', 0.9, 1.0, 0.9, 76]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0.7, 1.0, 0.7, 8]['VoLTE', 0.9, 1.0, 0.9, 9]['Video', 1.0, 1.0, 1.0, 15]['URLLC', 0.6, 1.0, 0.6, 7]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.8, 1.0, 0.8, 158]['URLLC', 0.7, 1.0, 0.7, 1]['VoLTE', 0.9, 1.0, 0.9, 102]['Video', 1.0, 1.0, 1.0, 118]['URLLC', 0.9, 1.0, 0.9, 10]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 0.8, 1.0, 0.8, 52]['URLLC', 0.9, 1.0, 0.9, 40]['VoLTE', 0.8, 1.0, 0.8, 10]['Video', 0.6, 1.0, 0.6, 8]['URLLC', 1.0, 1.0, 1.0, 117]['VoLTE', 0.9, 1.0, 0.9, 89]['Video', 0.6, 1.0, 0.6, 13]['URLLC', 1.0, 1.0, 1.0, 119]['VoLTE', 0.9, 1.0, 0.9, 11]['Video', 0.7, 1.0, 0.7, 15]['URLLC', 1.0, 1.0, 1.0, 75]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 0.8, 1.0, 0.8, 7]['URLLC', 0.7, 1.0, 0.7, 4]['VoLTE', 0.9, 1.0, 0.9, 111]['Video', 0.8, 1.0, 0.8, 10]['URLLC', 0.6, 1.0, 0.6, 6]['VoLTE', 1.0, 1.0, 1.0, 23]['Video', 0.9, 1.0, 0.9, 19]['URLLC', 0.8, 1.0, 0.8, 4]['VoLTE', 0.9, 1.0, 0.9, 14]['Video', 0.9, 1.0, 0.9, 174]['URLLC', 0.8, 1.0, 0.8, 4]['VoLTE', 0.9, 1.0, 0.9, 89]['Video', 1.0, 1.0, 1.0, 106]['URLLC', 1.0, 1.0, 1.0, 40]['VoLTE', 0.9, 1.0, 0.9, 8]['Video', 0.8, 1.0, 0.8, 47]['URLLC', 1.0, 1.0, 1.0, 99]['VoLTE', 0.9, 1.0, 0.9, 15]['Video', 0.9, 1.0, 0.9, 23]['URLLC', 1.0, 1.0, 1.0, 121]['VoLTE', 0.9, 1.0, 0.9, 96]['Video', 0.9, 1.0, 0.9, 12]['URLLC', 1.0, 1.0, 1.0, 63]['VoLTE', 0.9, 1.0, 0.9, 12]['Video', 0.8, 1.0, 0.8, 169]['URLLC', 0.7, 1.0, 0.7, 2]['VoLTE', 0.8, 1.0, 0.8, 14]['Video', 1.0, 1.0, 1.0, 105]['URLLC', 0.6, 1.0, 0.6, 3]['VoLTE', 0.9, 1.0, 0.9, 85]['Video', 0.8, 1.0, 0.8, 19]['URLLC', 0, 1.0, 0.6, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 11]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 11]['Video', 0.8, 1.0, 0.8, 9]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 103]['Video', 0.9, 1.0, 0.9, 14]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 1.0, 1.0, 1.0, 10]['Video', 0.8, 1.0, 0.8, 148]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 9]['Video', 1.0, 1.0, 1.0, 61]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 66]['Video', 0.8, 1.0, 0.8, 15]['URLLC', 0, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 11]['Video', 0.7, 1.0, 0.7, 12]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 0.7, 1.0, 0.7, 9]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 107]['Video', 1.0, 1.0, 1.0, 13]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 10]['Video', 0.8, 1.0, 0.8, 183]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 10]['Video', 1.0, 1.0, 1.0, 99]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 69]['Video', 0.8, 1.0, 0.8, 33]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.9, 1.0, 0.9, 21]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.8, 1.0, 0.8, 13]['Video', 0.9, 1.0, 0.9, 9]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 70]['Video', 0.8, 1.0, 0.8, 148]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 11]['Video', 1.0, 1.0, 1.0, 78]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 13]['Video', 0.8, 1.0, 0.8, 14]['URLLC', 1, 0.0, 1.0, 0]['VoLTE', 0.9, 1.0, 0.9, 99]"


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







































