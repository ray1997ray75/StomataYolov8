import seaborn as sns
import matplotlib.pyplot as plt
#pandas
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt

import os


Resultscsvlist = [os.path.join('./T32Vallabels',x) for x in os.listdir('./T32Vallabels') if x[-3:] == 'csv'  ]

sns.set_theme(style="white")


for i in Resultscsvlist:
    df = pd.read_csv(i)
    output =sns.relplot(
        data = df,
        x = "center_x" , y ="center_y"
    )
    output.savefig("out.png")
    #df = sns.load_dataset( i )
    





