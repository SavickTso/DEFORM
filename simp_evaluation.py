import json

import numpy as np

with open("loss_total_150-235.json", "r") as f:
    loss_list = json.load(f)

loss_list = np.array(loss_list)

print(np.average(loss_list[:, :1]))
print(np.average(loss_list[:, :5]))
print(np.average(loss_list[:, :10]))
print(np.average(loss_list[:, :15]))
print(np.average(loss_list[:, :20]))
