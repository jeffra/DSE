from os import name
import re
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

if len(sys.argv) == 1:
    exit()

name = ""

# matplotlib.use('GTK3Agg')

for filepath in sys.argv[1:]:
    iters = []
    losses = []
    loss_scales = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            match = re.search(r'iteration( *)(\d+)\/', line)
            if not match:
                continue
            iters.append(int(match.group(2)))

            match = re.search(r'lm loss: (.*?) ', line)
            if not match:
                print("{}".format(line))
            losses.append(float(match.group(1)))

            match = re.search(r'loss scale: (.*?) ', line)
            loss_scales.append(float(match.group(1)))

    n = Path(filepath).stem
    plt.plot(iters, losses, label=n)
    name += n + "-"

plt.locator_params(axis='x', nbins=10)
plt.xlabel('iteration')
plt.ylabel('loss')
name = name[:-1]
plt.title(name)
plt.legend()
plt.savefig(name)
# plt.show()
