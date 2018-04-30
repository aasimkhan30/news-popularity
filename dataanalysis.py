import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/data2.csv")
corrdict = {}
for column in df:
    if(df[column].corr(df[' shares']) <= 0.8):
        corrdict[column] = df[column].corr(df[' shares'])


corrdict = sorted(corrdict.items(), key=lambda x: x[1])

print(corrdict)
sortcorr = {}
for i in corrdict:
    sortcorr[i[0]] = i[1]

print(len(sortcorr))
plt.bar(range(len(sortcorr)), list(sortcorr.values()), align='center')
plt.xticks(range(len(sortcorr)), list(sortcorr.keys()), rotation='vertical')

plt.tight_layout()
plt.show()

