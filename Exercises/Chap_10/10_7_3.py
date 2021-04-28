import matplotlib.pyplot as plt
import numpy as np
import random

labels = ["lbl1", "lbl2"]

n_lbl1 = n_lbl2 = 0

while n_lbl1 == 0 or n_lbl2 == 0:
    obs = [(1,4,random.choice(labels)), (1,3,random.choice(labels)), (0,4,random.choice(labels)), (5,1,random.choice(labels)), (6,2,random.choice(labels)), (4,0,random.choice(labels))]

    x = list(map(lambda x: x[0], obs))
    y = list(map(lambda x: x[1], obs))

    n_lbl1 = len(list(filter(lambda x: x[2] == "lbl1", obs)))
    n_lbl2 = len(list(filter(lambda x: x[2] == "lbl2", obs)))


lbl1_centroid_x = lbl1_centroid_x_original = sum([x[0] for x in filter(lambda y: y[2] == "lbl1", obs)]) / n_lbl1
lbl1_centroid_y = lbl1_centroid_y_original = sum([x[1] for x in filter(lambda y: y[2] == "lbl1", obs)]) / n_lbl1

lbl2_centroid_x = lbl2_centroid_x_original = sum([x[0] for x in filter(lambda y: y[2] == "lbl2", obs)]) / n_lbl2
lbl2_centroid_y = lbl2_centroid_y_original = sum([x[1] for x in filter(lambda y: y[2] == "lbl2", obs)]) / n_lbl2

_, ax = plt.subplots()
plt.scatter(x, y)
plt.scatter([lbl1_centroid_x_original, lbl2_centroid_x_original], [lbl1_centroid_y_original, lbl2_centroid_y_original], marker="x")

for o in obs:
    ax.annotate(o[2], (o[0], o[1]))

ax.annotate("lbl1_centroid", (lbl1_centroid_x, lbl1_centroid_y))
ax.annotate("lbl2_centroid", (lbl2_centroid_x, lbl2_centroid_y))

plt.show()

np_centroid_lbl1 = np.array((lbl1_centroid_x, lbl1_centroid_y))
np_centroid_lbl2 = np.array((lbl2_centroid_x, lbl2_centroid_y))

new_obs = list.copy(obs)
has_changes = True

while has_changes:
    obs = list.copy(new_obs)
    has_changes = False

    lbl1_centroid_x = sum([x[0] for x in filter(lambda y: y[2] == "lbl1", obs)]) / n_lbl1
    lbl1_centroid_y = sum([x[1] for x in filter(lambda y: y[2] == "lbl1", obs)]) / n_lbl1

    lbl2_centroid_x = sum([x[0] for x in filter(lambda y: y[2] == "lbl2", obs)]) / n_lbl2
    lbl2_centroid_y = sum([x[1] for x in filter(lambda y: y[2] == "lbl2", obs)]) / n_lbl2

    np_centroid_lbl1 = np.array((lbl1_centroid_x, lbl1_centroid_y))
    np_centroid_lbl2 = np.array((lbl2_centroid_x, lbl2_centroid_y))

    for i, o in enumerate(obs):
        np_obs = np.array((o[0], o[1]))
        dist_lbl1 = np.linalg.norm(np_centroid_lbl1 - np_obs)
        dist_lbl2 = np.linalg.norm(np_centroid_lbl2 - np_obs)

        if dist_lbl1 <= dist_lbl2 and o[2] != "lbl1":
            has_changes = True
            new_obs[i] = (o[0], o[1], "lbl1")
        elif dist_lbl1 > dist_lbl2 and o[2] != "lbl2":
            has_changes = True
            new_obs[i] = (o[0], o[1], "lbl2")

_, ax = plt.subplots()
plt.scatter(x, y)
plt.scatter([lbl1_centroid_x, lbl2_centroid_x], [lbl1_centroid_y, lbl2_centroid_y], marker="x")

for o in obs:
    ax.annotate(o[2], (o[0], o[1]))

ax.annotate("lbl1_centroid", (lbl1_centroid_x, lbl1_centroid_y))
ax.annotate("lbl2_centroid", (lbl2_centroid_x, lbl2_centroid_y))

plt.show()

np_centroid_lbl1 = np.array((lbl1_centroid_x, lbl1_centroid_y))
np_centroid_lbl2 = np.array((lbl2_centroid_x, lbl2_centroid_y))