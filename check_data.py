from scipy.io import loadmat

d = loadmat("SNLS80mV.mat")
print([k for k in d.keys() if not k.startswith("__")])
for k in d:
    if not k.startswith("__"):
        print(k, d[k].shape)