#%% HDR: npy concatenation | start by start
import numpy as np
import os
import pandas as pd


#%% Settings
conn = 'full'  # ['full', 'sparse']

std_X = 'z_score'
std_t = 'max_min'

year_s = 2010
year_f = 2016
period = f'{year_s}_{year_f}'

# years = pd.Series(np.arange(year_s, year_f+1))
year  = 2016

n_attrs  = 97      # number of attributes
n_hidden = 12      # number of neurons in the hidden layer
n_iters  = 100000  # number of iterations per start
l_rate   = 1       # learning rate  (0 < l_rate <= 1)

modulo = 10000

#%% Set input/output folders
settings = f'{conn}/{std_X}__{std_t}/{period}/{year}/' \
           f'{n_iters}i_{n_hidden}h_{l_rate}l'

data_in  = f'../data/HDR_3_nn_1a_init_training/{settings}'
data_out = f'../data/HDR_3_nn_2a_concat_npy/{settings}'

if not os.path.exists(data_out):
    os.makedirs(data_out)


#%% Concatenate files
files = sorted(os.listdir(data_in))


#%% Bias input-hidden layer
# B_x = np.stack([np.load(f'{data_in}/{file}')
#                 for file in files
#                 if 'B_x_res.npy' in file])
# np.save(f'{data_out}/B_x', B_x)
# print('B_x:', B_x.shape)
# del(B_x)


#%% Bias hidden-output layer
# B_h = np.stack([np.load(f'{data_in}/{file}')
#                 for file in files
#                 if 'B_h_res.npy' in file])
# np.save(f'{data_out}/B_h', B_h)
# print('B_h:', B_h.shape)
# del(B_h)


#%% Weights of the input layer
# W_x = np.stack([np.load(f'{data_in}/{file}')
#                 for file in files
#                 if 'W_x_res.npy' in file])
# np.save(f'{data_out}/W_x', W_x)
# print('W_x:', W_x.shape)
# del(W_x)


#%% Weights of the hidden layer
# W_h = np.stack([np.load(f'{data_in}/{file}')
#                 for file in files
#                 if 'W_h_res.npy' in file])
# np.save(f'{data_out}/W_h', W_h)
# print('W_h:', W_h.shape)
# del(W_h)


#%% Outputs
y = np.stack([np.load(f'{data_in}/{file}')
              for file in files
              if 'y_res.npy' in file])
np.save(f'{data_out}/y', y)
print('y:  ', y.shape)


#%% Mutual countries HDR (2010-2016)
cntr = pd.read_csv(f'../data/HDR_mutual_attributes_and_countries/'
                   f'countries_{period}.csv', header=None)
cntr = list(cntr[0])

print('Total countries:', len(cntr))

#%% Calculate loss function; Output data to csv format
t = pd.read_csv(f'../data/HDR_2a_{std_t}_year/{year}.csv', index_col='Country')
t = t.loc[cntr]  # mutual countries
t = t[t.notna()['Human Development Index (HDI)'] == True]
t = t['Human Development Index (HDI)'].values  # target values

C = np.zeros((y.shape[0], y.shape[2]))

for s in range(y.shape[0]):
    C[s, :] = np.average((np.subtract(y[s, :, :].T, t).T ** 2) / 2, axis=0)

np.save(f'{data_out}/C', C)
print('C:  ', C.shape)
del(C)
del(y)


#%% Multiply the weights between the layers; Output data to csv format
prefixes = sorted(list(set([file.split('_')[0] for file in files])))
n_starts = len(prefixes)

W = np.ndarray((n_starts, n_attrs, n_iters))

for prefix in prefixes:
    s = prefixes.index(prefix)
    print(f'========== {s+1}/{n_starts} ==========')

    W_x = np.load(f'{data_in}/{prefix}_W_x_res.npy')
    W_h = np.load(f'{data_in}/{prefix}_W_h_res.npy')
    # print('W_x:', W_x.shape)
    # print('W_h:', W_h.shape)

    for i in range(n_iters):
        if (i+1) % modulo == 0:
            print(f'{year} | {s+1}/{n_starts} | Iteration {i+1}/{n_iters}...',
                  end=' ')
            
        W[s, :, i] = np.dot(W_x[:, :, i], W_h[:, i])
        
        if (i+1) % modulo == 0:
            print('Done!')
    print('\n')
    
print('Saving weights...', end=' ')
np.save(f'{data_out}/W', W)
print('Done!')
print('W:  ', W.shape)
del(W)
