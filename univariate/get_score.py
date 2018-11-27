import pandas as pd
import math
import sys
df = pd.read_csv(sys.argv[1])

for horizon in [3, 6, 12, 24]:
    mae, rmse = 1e9, 1e9

    for timestep in df.timestep.unique():
        for n_hidden in df.n_hidden.unique():
                tmpdf =  df[(df.timestep == timestep) & (df.n_hidden == n_hidden) & (df.horizon == horizon)]
                if tmpdf['mae'].mean() < mae:
                    mae = tmpdf['mae'].mean()
                    rmse = math.sqrt(tmpdf['mse'].mean())

                    x = timestep
                    y = n_hidden
    print('====== horizon {} ====='.format(horizon))
    print('mae', mae, 'rmse', rmse)
    print('best params:', 'timestep', x, 'n_hidden', y)
    print('=======================')
    
