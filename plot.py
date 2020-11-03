import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook
import re

with open('./log.txt', 'r') as f:
    data = f.read()

    #print(data)

losses = re.findall('ACC:\[ \d.\d*', data)
print(losses[0][6:])

losses = re.findall('Loss:\s\s\d.\d*', data)
print(losses)
accs = re.findall(' ACC:\[ \d.\d*', data)

loss_values = []
acc_values = []

for item in losses:
    loss_values.append(float(item[7:]))
for item in accs:
    acc_values.append(float(item[6:]))

plt.figure(figsize = (15, 8))
plt.subplot(121)
plt.title('Implementation Result', size=20)
plt.plot(range(len(loss_values)),loss_values)
plt.xlabel('Iter', size=15)
plt.ylabel('Loss', size=15)

plt.subplot(122)
plt.title('Implementation Result', size = 20)
plt.plot(range(len(acc_values)),acc_values)
plt.xlabel('Iter', size = 15)
plt.ylabel('ACC', size = 15)
plt.show()
