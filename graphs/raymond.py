import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0.01

episode = []
average_score = []
current_average = 0

fo = open("cms_2_alt.txt", "r")

for line in fo.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = float(re.findall(r'[0-9.]+', search.group())[9])
        if e not in episode:
            episode.append(e)
            average_score.append(current_average/7)
            current_average = 0
        avg = float(re.findall(r'[0-9.]+', search.group())[12])
        current_average += avg

line_up, = plt.plot(episode,average_score,label="Alternating Marines")

plt.xlabel('Episodes')
plt.ylabel('Average score')

episode_ca = []
average_score_ca = []
current_average_ca = 0

f1 = open("cms_2_alt_control_another.txt", "r")

for line in f1.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = float(re.findall(r'[0-9.]+', search.group())[9])
        if e not in episode_ca and e <= 4176:
            episode_ca.append(e)
            average_score_ca.append(current_average_ca/7)
            current_average_ca = 0
        avg = float(re.findall(r'[0-9.]+', search.group())[12])
        current_average_ca += avg

line_down, = plt.plot(episode_ca,average_score_ca, label="Grouped Marines")

episode_mc = []
average_score_mc = []
current_average_mc = 0

f2 = open("sc2g_a2c_multimovement_cms_1.txt", "r")

for line in f2.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = float(re.findall(r'[0-9.]+', search.group())[9])
        if e not in episode_mc:
            episode_mc.append(e)
            average_score_mc.append(current_average_mc/7)
            current_average_mc = 0
        avg = float(re.findall(r'[0-9.]+', search.group())[12])
        current_average_mc += avg
for x in range(2146, 4176):
    episode_mc.append(x)
    average_score_mc.append(10)

line_r, = plt.plot(episode_mc,average_score_mc, label="Two Pairs of Co-ordinates")
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, handles=[line_up, line_down, line_r])

plt.savefig("raymond.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

fo.close()
