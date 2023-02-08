import io
import pandas as pd
import numpy as np
import scipy.stats as st
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns


def show_info(dataframe):
    buf = io.StringIO()
    dataframe.info(buf=buf)
    s = buf.getvalue()
    ss = s.splitlines()[3:-2]
    ss = [re.split('\s\s+', line.strip()) for line in ss]
    info = pd.DataFrame(data=ss[2:], columns=ss[0])
    info.drop('#', axis=1, inplace=True)

    desc = dataframe.describe().T.reset_index()
    return pd.concat([desc, info[['Dtype', 'Non-Null Count']]], axis=1).style.background_gradient(
        subset=['mean', 'std', '50%', 'min', 'max'], cmap='Blues'
    )

def get_uniques(train, original, feats):
    uniques = []
    for f in feats:
        item = {"feature": f}
        count = len(train[f].unique())
        item['train'] = count
        count_orig = len(original[f].unique())
        item['original'] = count_orig
        if count < 10:
            item['values'] = train[f].unique().tolist()
        elif count < train.shape[0]:
            item['values'] = train[f].unique().tolist()[:10] + ['...']
        else:
            item['values'] = 'all are unique'
        if count_orig < 10:
            item['orig_values'] = original[f].unique().tolist()
        elif count_orig < original.shape[0]:
            item['orig_values'] = original[f].unique().tolist()[:10] + ['...']
        else:
            item['orig_values'] = 'all are unique'

        return uniques.append(item)

