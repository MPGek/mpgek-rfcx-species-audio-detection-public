import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv(r'd:\Projects\Kaggle\rfcx-species-audio-detection\data\runs_server\20210116_best' +
                   r'\sub_gf_best_e_id_0_10299_210116_1245_MEAN_Lwlrap_0.8752_f2_0.2080_LB0.867.csv')
                   # r'\sub_gf_best_e_id_3_10299_210116_1337_MEAN_Lwlrap_0.8924_f2_0.1996_LB0.850.csv')

data_np = data.values.tolist()
data_np = np.array(data_np)  # type:np.ndarray
data_np = data_np[:, 1:].astype(np.float)
data_np = data_np.reshape([-1])

sns.histplot(data_np, bins=50)
plt.show()
