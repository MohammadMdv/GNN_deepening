# ploting the accuracy of the model with epoch_num as x-axis and accuracy as y-axis
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accs(accs, epoch_num):
    plt.plot(np.arange(epoch_num), accs)
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')
    plt.show()

# Reading the accuracy data from Cora.txt
accs = []
with open(os.path.join('test_accs', 'Cora.txt'), 'r') as f:
    for line in f:
        accs.append(float(line.strip()))
# Plotting the data
plot_accs(accs, len(accs))


