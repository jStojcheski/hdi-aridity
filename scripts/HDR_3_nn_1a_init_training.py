# HDR: Neural Network Initial Training
import numpy as np
import pandas as pd
import os
import time

#%% settings
std_X = 'z_score'  # standardization method for input values (X)
std_t = 'max_min'  # standardization method for output values (t)

conn = 'full'  # ['full', 'sparse']

year_s = 2010
year_f = 2016
period = f'{year_s}_{year_f}'

year = 2016  # [year_s, year_f]

#%% Set input/output folder
data_in_X = f'../data/HDR_2a_{std_X}_year'
data_in_t = f'../data/HDR_2a_{std_t}_year'

data_mt  =  '../data/HDR_mutual_attributes_and_countries'
data_out = f'../data/HDR_3_nn_1a_init_training/{conn}/' \
           f'{std_X}__{std_t}/{period}/{year}'

if not os.path.exists(data_out):
    os.makedirs(data_out)

print(f'{conn}/{std_X}__{std_t}/{period}/{year}', end='\n\n')

#%% Mutual attributes (2010-2016)
attr = pd.read_csv(f'{data_mt}/attributes_{period}.csv', header=None)
attr = list(attr[0])

print('Total attributes:', len(attr))

#%% Mutual countries (2010-2016)
cntr = pd.read_csv(f'{data_mt}/countries_{period}.csv', header=None)
cntr = list(cntr[0])

print('Total countries:', len(cntr))

#%% Read input data | Input values
X = pd.read_csv(f'{data_in_X}/{year}.csv', index_col='Country')

X = X[attr]      # mutual attributes
X = X.loc[cntr]  # mutual countries

X = X[X.notna()['Human Development Index (HDI)'] == True]
print(f'{len(cntr) - X.shape[0]} countries are missing HDI value.')

X.replace(0, 1e-9)  # 1/(1,000,000,000)
X = X.fillna(0)    # 0 doesn't contribute to weights

X = X.drop('Human Development Index (HDI)', axis=1)  # input values
# X[::10]

#%% Read input data | Target values
t = pd.read_csv(f'{data_in_t}/{year}.csv', index_col='Country')

t = t.loc[cntr]  # mutual countries
t = t[t.notna()['Human Development Index (HDI)'] == True]
t = t['Human Development Index (HDI)']  # target values

print(f'{len(cntr) - t.shape[0]} countries are missing AI value.')

# t[::10]

#%% Get list and size of all dimensions
dim_list = [file for file in sorted(os.listdir('../data/HDR_0'))
            if os.path.isdir(f'../data/HDR_0/{file}')]
dim_len = dict()

for dim in dim_list:
    dim_len[dim] = X.filter(regex='\[{}\].*'.format(dim)).shape[1]
    print('-', dim, f'({dim_len[dim]})')
print()


#%% Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))


def clean_weights(W, A):
    if W.shape[0] != A.shape[0] or W.shape[1] != A.shape[1]:
        print(W.shape, '!=', A.shape)
        return

    else:
        W_temp = np.zeros(W.shape)
        c = 0
        
        for i in range(n_hidden):
            for j in range(dim_len[dim_list[i]]):
                W_temp[j+c, i] = W[j+c, i]
            c += dim_len[dim_list[i]]

    return W_temp


def get_attr_weights(W):
    W_arr = np.zeros(n_input)
    c = 0

    for i in range(n_hidden):
        for j in range(dim_len[dim_list[i]]):
            W_arr[j + c] = W[j + c, i]
        c += dim_len[dim_list[i]]

    return W_arr


#%% Neural network initial training
# settings
n_start = 1       # number of random starts
n_iters = 100000  # number of iterations per start
l_rate  = 1       # learning rate  (0 < l_rate <= 1)

n_sampl = X.shape[0]  # number of items/samples
n_input = X.shape[1]  # number of attributes (neurons in input layer)
n_hidden = len(dim_list)  # number of dimensions (neurons in hidden layer)

low = -0.3  # lower bound of uniform distribution
high = 0.3  # upper bound of uniform distribution

#%% Iterate
# set output folder
nd_out = f'{data_out}/{n_iters}i_{n_hidden}h_{l_rate}l'

if not os.path.exists(nd_out):
    os.makedirs(nd_out)

# initialize & train the neural network (sparsely-connected)
if conn == 'sparse':  # update only the weights between attributes and their corresponding dimension
    W_x_res = np.zeros((n_input, n_iters))
    B_x_res = np.zeros((n_hidden, n_iters))  # bias neuron for each dimension

# initialize & train the neural network (fully-connected)
if conn == 'full':  # update all the weights between input and hidden layer
    W_x_res = np.zeros((n_input, n_hidden, n_iters))
    B_x_res = np.zeros(n_iters)  # only one bias neuron

W_h_res = np.zeros((n_hidden, n_iters))
B_h_res = np.zeros(n_iters)
y_res = np.zeros((n_sampl, n_iters))

tim = time.time()

for s in range(n_start):
    print(f'===== RANDOM START {s+1} =====')
    
    iter_tim = time.time()
    
    if conn == 'sparse':  # (sparsely-connected network)
        # initialize the biases of the input layer
        B_x = np.zeros((n_hidden, 1))  # biases of the input layer
        
        # initialize the weights between the input and the hidden layer
        W_x = np.zeros(
            (n_input, n_hidden))  # weights between the input and the hidden layer
        A_x = np.zeros(W_x.shape, dtype=bool)  # binary matrix of used weights
        
        c = 0
        for i in range(n_hidden):
            for j in range(dim_len[dim_list[i]]):
                # print(j + c, i)
                W_x[j + c, i] = np.random.uniform(low=low, high=high)
                A_x[j + c, i] = True
            c += dim_len[dim_list[i]]
        
        # initialize the weights between the hidden and the output layer
        W_h = np.zeros(
            (n_hidden, 1))  # weights between the hidden and the output layer
        for i in range(n_hidden):
            if dim_len[dim_list[i]] > 0:
                W_h[i, 0] = np.random.uniform(low=low, high=high)
    
    if conn == 'full':  # (fully-connected network)
        # initialize the bias of the input layer
        B_x = np.zeros((1, 1))  # bias of the input layer
        
        # initialize the weights between the input and the hidden layer
        W_x = np.random.uniform(low=low, high=high, size=(n_input, n_hidden))
        
        # initialize the weights between the hidden and the output layer
        W_h = np.random.uniform(low=low, high=high, size=(
            n_hidden, 1))  # weights between the hidden and the output layer
    
    # initialize the bias of the hidden layer
    B_h = np.zeros((1, 1))  # bias of the hidden layer
    
    # start training...
    for i in range(n_iters):
        # shuffle items
        perm_index = np.random.permutation(n_sampl)
        # X = X.reindex(perm_index)
        # t = t.reindex(perm_index)
        
        y = np.zeros(t.shape)  # create output-values vector
        
        for j in perm_index:
            # feed-forward
            X_i = np.matrix(X.iloc[j]).T  # input layer values
            z_h = np.dot(W_x.T, X_i) + B_x  # non-activated hidden-layer values
            X_h = sigmoid(z_h)  # activated hidden-layer values
            z_y = np.dot(W_h.T, X_h) + B_h  # non-activated output layer values
            y[j] = sigmoid(z_y)  # activated output-layer values
            
            # calculate derivatives
            der_B_h = (y[j] - t[j]) * sigmoid_derivative(z_y)  # derivative of the bias of the hidden layer
            der_W_h = np.multiply(der_B_h,X_h)  # derivative of the weights between the hidden layer and the output layer
            der_B_x = np.multiply(der_B_h, np.multiply(W_h, sigmoid_derivative(z_h)))  # derivative of the biases of the input layer
            der_W_x = (X_i * der_B_x.T)  # derivative of the weights between the input layer and the hidden layer
            
            # backpropagate
            B_h -= np.multiply(l_rate, der_B_h)  # backpropagation on the bias of the hidden layer
            W_h -= np.multiply(l_rate, der_W_h)  # backpropagation on the weights between the hidden layer and the output layer
            W_x -= np.multiply(l_rate, der_W_x)  # backpropagation on the weights between the input layer and the hidden layer
            
            if conn == 'sparse':
                B_x -= np.multiply(l_rate, der_B_x)  # backpropagation on the biases of the input layer
            
            if conn == 'full':
                B_x -= np.multiply(l_rate, der_B_x).sum()  # backpropagation on the biases of the input layer
            
            # keep only the values of interest
            # (it pushes the neural network the update only the weights
            # between the attributes and their corresponding dimensions)
            if conn == 'sparse':
                W_x = clean_weights(W_x, A_x)
        
        # calculate loss
        C = np.average(((y - t) ** 2) / 2)
        
        # save the weights, biases & outputs of the s-th start in the i-th iteration
        if conn == 'sparse':  # (sparsely-connected network)
            W_x_res[:, i] = get_attr_weights(
                W_x)  # get the weights of the attributes
            B_x_res[:, i] = B_x.flatten()
        
        if conn == 'full':  # (fully-connected network)
            W_x_res[:, :, i] = W_x
            B_x_res[i] = B_x
        
        W_h_res[:, i] = W_h.flatten()
        B_h_res[i] = B_h
        y_res[:, i] = y
        
        # if (i + 1) % max((n_iters // 100), 1) == 0 or i == 0:
        if (i + 1) % 10 == 0 or i == 0:
            print(f'[{year} | {s+1}/{n_start}] Iteration {i+1:>7d} | Loss = {C:.15f}')

    # output np.ndarray(s) to npy format
    now = ''.join([f'{item:02d}' for item in time.localtime()[:]])

    print('Saving W_x_res ...', end=' ')
    np.save(f'{nd_out}/{now}_W_x_res', W_x_res)
    print('Done!')

    print('Saving W_x_res ...', end=' ')
    np.save(f'{nd_out}/{now}_B_x_res', B_x_res)
    print('Done!')

    print('Saving W_x_res ...', end=' ')
    np.save(f'{nd_out}/{now}_W_h_res', W_h_res)
    print('Done!')

    print('Saving W_x_res ...', end=' ')
    np.save(f'{nd_out}/{now}_B_h_res', B_h_res)
    print('Done!')

    print('Saving W_x_res ...', end=' ')
    np.save(f'{nd_out}/{now}_y_res', y_res)
    print('Done!')

    print('All files are successfully saved!')
    
    print(f'\nRANDOM START {s+1}: {n_iters} iterations done in '
          f'{int(time.time() - iter_tim)} seconds.', end='\n\n')

print(f'The whole process took {int(time.time() - tim)} seconds.', end='\n\n')


#%% #### Linux run
# <code>for x in {1..5}; do (python HDR_3_nn_1a_init_training.py > run_$x.log) & done</code>
# <code>for x in {1..5}; do (python HDR_3_nn_1a_init_training.py > run_$(date +%s%N).log) & done</code>
