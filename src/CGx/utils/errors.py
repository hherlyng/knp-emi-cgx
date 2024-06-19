import numpy  as np
import pandas as pd

# Define variable names
vars = ['Na_i', 'Na_e', 'K_i', 'K_e', 'Cl_i', 'Cl_e', 'phi_i', 'phi_e']

## 2D L2 error norms
data_2D = np.array([
                [0.006036, 0.018953, 0.006393, 0.006682, 0.012358, 0.025353, 0.088013, 0.038644],
                [0.001483, 0.005382, 0.002071, 0.001972, 0.003514, 0.007111, 0.024922, 0.011732],
                [0.000356, 0.001394, 0.000571, 0.000526, 0.000912, 0.001831, 0.006415, 0.003114],
                [0.000081, 0.000349, 0.000155, 0.000140, 0.000230, 0.000461, 0.001604, 0.000801],
                [0.000015, 0.000086, 0.000048, 0.000043, 0.000058, 0.000114, 0.000389, 0.000212],
                [0.000009, 0.000022, 0.000021, 0.000019, 0.000014, 0.000028, 0.000090, 0.000066],
                [0.000012, 0.000013, 0.000015, 0.000014, 0.000003, 0.000006, 0.000034, 0.000033]
            ])
## 3D L2 error norms
data_3D = np.array([
                [0.011005, 0.055599, 0.012447, 0.018460, 0.022377, 0.074032, 0.099216, 0.042177],
                [0.004914, 0.027820, 0.005873, 0.009515, 0.010617, 0.037258, 0.052013, 0.028375],
                [0.001587, 0.008564, 0.001870, 0.002930, 0.003422, 0.011433, 0.017401, 0.011334],
                [0.000432, 0.002266, 0.000505, 0.000773, 0.000923, 0.003013, 0.004729, 0.003220],
                [0.000113, 0.000575, 0.000127, 0.000196, 0.000236, 0.000763, 0.001209, 0.000832]
            ])

# Create dataframes with L2 error norms
df_errors_2D = pd.DataFrame(data = data_2D, columns = vars)
df_errors_3D = pd.DataFrame(data = data_3D, columns = vars)

# Create dataframes for the convergence rates
df_rates_2D = df_errors_2D.copy(deep = True)
df_rates_2D.drop(df_rates_2D.tail(1).index, inplace=True) # Drop one row since number of rates is number of norms less one

df_rates_3D = df_errors_3D.copy(deep = True)
df_rates_3D.drop(df_rates_3D.tail(1).index, inplace=True) # Drop one row since number of rates is number of norms less one

for col in df_errors_2D.columns:
    df_rates_2D[col] = -(1/np.log(2)) * np.log(df_errors_2D[col][1:].values / df_errors_2D[col][:-1].values)
    df_rates_3D[col] = -(1/np.log(2)) * np.log(df_errors_3D[col][1:].values / df_errors_3D[col][:-1].values)

print("2D L2 erorr norm rates:")
print(df_rates_2D)
print("\n3D L2 error norm rates:")
print(df_rates_3D)