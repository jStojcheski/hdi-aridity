%% wsbm
addpath(genpath('!tools/WSBM_v1.2'))

w_distr   = 'exp';     % ['exp',  'norm']
alpha     = 0.5;       % 0 = pure WSBM

for year=2010:2016
    l1 = csvread(sprintf('../data/HDR_4a_graph_formation/mean/l1_%d.csv', ...
                 year), 1, 1);
    
%    best_le = -inf;
%    best_k  = -inf;
%    best_l  = -inf;
    start = 5;
    
    ks = start:5:20;
    siz_tmp = size(l1);

    labels_out = zeros(siz_tmp(1), length(ks));
    logevd_out = zeros(1, length(ks));
    
    index = 1;

    for k=ks
        [labels, model] = wsbm(l1, k, 'W_Distr', w_distr, 'alpha', alpha);
        logEvidence = model.Para.LogEvidence;

%         if logEvidence >= best_le
%             best_le = logEvidence;
%             best_k  = k;
%             best_l  = labels;
%         end

        labels_out(:, index) = labels;
        logevd_out(1, index) = logEvidence;
        index = index + 1;
    end

    csvwrite(sprintf('../data/HDR_4c_wsbm/00_raw/full/%s_0.5/%d_labels_%d.csv', ...
             w_distr, year, start), labels_out);
    csvwrite(sprintf('../data/HDR_4c_wsbm/00_raw/full/%s_0.5/%d_logevidence_%d_.csv', ...
             w_distr, year, start), logevd_out);
end