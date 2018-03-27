import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

fig_size = plt.rcParams["figure.figsize"]

fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

episode = []
average_score = []
current_average = 0

plt.xlabel('Episodes')
plt.ylabel('Average score')

f1 = open("grouping_random_t0.txt", "r")

episode_gr = []
score_gr = []

for line in f1.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = int(re.findall(r'[0-9.]+', search.group())[8])
        if e <= 4176:
            episode_gr.append(e)
            score= int(re.findall(r'[0-9.]+', search.group())[13])
            score_gr.append(score)


line_up, = plt.plot(episode_gr,score_gr,label="Grouped Random")

f2 = open("grouping_script_t0.txt", "r")

episode_gs = []
score_gs = []

for line in f2.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = int(re.findall(r'[0-9.]+', search.group())[8])
        if e <= 4176:
            episode_gs.append(e)
            score= int(re.findall(r'[0-9.]+', search.group())[13])
            score_gs.append(score)


line_down, = plt.plot(episode_gs,score_gs, label="Grouped Scripted")
#
f3 = open("individual_random_ma_t0.txt", "r")

episode_ir = []
score_ir = []

for line in f3.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = int(re.findall(r'[0-9.]+', search.group())[8])
        if e <= 4176:
            episode_ir.append(e)
            score= int(re.findall(r'[0-9.]+', search.group())[13])
            score_ir.append(score)


line_left, = plt.plot(episode_ir,score_ir, label="Individual Random")

f4 = open("individual_script_non_ma_t0.txt", "r")

episode_in = []
score_in = []

for line in f4.readlines():
    search = re.match(r'(.*)episode [0-9]+(.*)', line, re.M|re.I)
    if search:
        #episode number
        e = int(re.findall(r'[0-9.]+', search.group())[8])
        if e <= 4176:
            episode_in.append(e)
            score= int(re.findall(r'[0-9.]+', search.group())[13])
            score_in.append(score)


line_right, = plt.plot(episode_in,score_in, label="Individual Scripted")
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           handles=[line_right, line_down, line_up, line_left])
# plt.tight_layout()

plt.savefig("scripted.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

f1.close()
f2.close()
f3.close()
f4.close()
