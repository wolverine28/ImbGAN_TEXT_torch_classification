
import numpy as np
from IMDB_classification_func import run

IRs = np.array([10,50,100])
reps = np.array([0,1,2,3,4])

np.random.shuffle(reps)
np.random.shuffle(IRs)

for IR in IRs:
    for rep in reps:
        run(IR,rep,GPU_NUM=3)