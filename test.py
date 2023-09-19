import matplotlib.pyplot as plt

a = {"GCNII": 86.1, "APPNP": 83.63, "DAGNN": 82.7, "GPRGNN": 83.7, "JKnet": 78.7, "SGC": 60}

# plotting keys as X axis and values as Y axis in a bar graph
plt.bar(a.keys(), a.values())
plt.xlabel("Models trained on 1000 epochs on Cora dataset")
plt.ylabel("Accuracy")
plt.show()