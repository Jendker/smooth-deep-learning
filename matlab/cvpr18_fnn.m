% copyright @ Hao Shen, 2018
% Ref: Hao Shen, Towards a Mathematical Understanding of the 
%      Difficulty in Learning with Feedforward Neural Networks, 
%      IEEE-CVPR, 811–820, 2018.
% Aim: demonstrate local quadratic convergence properties of
%      the Approximate Newton's algorithm under exact learning
% Configuration: Bent identity used as the activation function in the
%      hidden layers.

function cvpr18_fnn
clc
warning off

% (1) load four region
load('four_region_v2.mat');
% number of training samples
pidx = randperm(1000, 10);
X_input = X_input(:,pidx);
Y_output = Y_output(:,pidx);

% (2) Setting paremeters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[d_input, n_sample] = size(X_input);% dimension of input, number of samples
d_output = size(Y_output,1);                          % dimension of output
% Configuration of MLP
n_neuron = [d_input,10,10,d_output];          	             % structure of MLP
n_layer = size(n_neuron,2)-1;                 % number of processing layers
n_variable = 0;                                  % number of total variabls
%     Initialize weights and bias
for lay_id = 1:n_layer
    % weight matrices in cell form
    WeightCell{lay_id} = randn(n_neuron(lay_id)+1, n_neuron(lay_id+1));
    n_variable = n_variable + (n_neuron(lay_id)+1)*n_neuron(lay_id+1);
end

% (3) Approximate Newton's algorithm
WeitCellList{1} = WeightCell;
for iterate = 1:1000
    % initializing gradient and hessian
    for lay_row = 1:n_layer                             % processing layers
        GradCell{lay_row} = zeros(n_neuron(lay_row)+1, n_neuron(lay_row+1));
    end
    Hessian = zeros(n_variable);                        % Hessian in matrix
    
    for sam_id = 1:n_sample
        % fetch a sample (Xin(:,sam_id)) and forward
        phin = [X_input(:,sam_id); 1];                  % phi input
        for lay_id = 1:(n_layer-1)
            temp = WeightCell{lay_id}'*phin;
            % Bent Identity activation function and its derivative
            pmet = 0.5 * ( temp ./ sqrt( temp.^2 + 1 ) ) + 1;
            temp = 0.5 * ( sqrt( temp.^2 + 1 ) - 1 ) + temp;

            PhiCell{lay_id} = temp;
            D2WCell{lay_id} = phin * pmet';           % derivative w.r.t. W
            D2FCell{lay_id} = WeightCell{lay_id} * diag( pmet );
            phin = [temp; 1];
        end
        PhiCell{n_layer} = WeightCell{n_layer}' * phin;
        D2WCell{n_layer} = repmat( phin, [1, n_neuron(n_layer+1)] );
        D2FCell{n_layer} = WeightCell{n_layer};
        
        % backward: error vector and gradient
        ErroCell{n_layer+1} = PhiCell{n_layer}-Y_output(:,sam_id); % error in measure
        for lay_id = n_layer:-1:1
            GradCell{lay_id} = GradCell{lay_id} + ...
                D2WCell{lay_id} * diag( ErroCell{lay_id+1} );
            temp = D2FCell{lay_id} * ErroCell{lay_id+1};
            ErroCell{lay_id} = temp(1:(size(temp,1)-1));
        end
        %
        JacoCell{n_layer} = eye(d_output);
        JacoTemp = JacoCell{n_layer};
        for lay_id = (n_layer-1):-1:1
            JacoCell{lay_id} = D2FCell{lay_id+1} * JacoTemp;
            JacoTemp = JacoCell{lay_id}(1:(size(JacoCell{lay_id})-1), :);
        end
       
        % compute the Hessian
        HessErro = eye(d_output);       	% Hessian in error: least squares
        for lay_rid = 1:n_layer
            for lay_cid = 1:n_layer
                PsiPsiT{lay_rid,lay_cid} = ...
                    JacoCell{lay_rid}*HessErro*(JacoCell{lay_cid})';
            end
        end
        % Hessian in cell
        for lay_rid = 1:n_layer
            for lay_cid = 1:n_layer
                for unt_rid = 1:n_neuron(lay_rid+1)
                    for unt_cid = 1:n_neuron(lay_cid+1)
                        TempCell{unt_rid, unt_cid} = ...
                            PsiPsiT{lay_rid,lay_cid}(unt_rid,unt_cid)*...
                            D2WCell{lay_rid}(:,unt_rid)*...
                            (D2WCell{lay_cid}(:,unt_cid))';
                    end
                end
                
                HessCell{lay_rid,lay_cid} = cell2mat(TempCell);
                clear('TempCell');
            end
        end
        
        Hessian = Hessian + cell2mat(HessCell);
    end
    
   	% vectorize gradient
    Gradient = [];
    for lay_id = 1:n_layer
        Gradient = [ Gradient; reshape(GradCell{lay_id},...
            [(n_neuron(lay_id)+1)*n_neuron(lay_id+1),1]) ];
    end
    Newton = (Hessian + (1e-8)*eye(n_variable)) \ Gradient;

    % Newton update
    alpha = 1;          % step size for exact learning
    Newton = alpha * Newton;
   	segs = 1;
    for lay_id = 1:n_layer
        shift = (n_neuron(lay_id)+1)*n_neuron(lay_id+1);
        WeightCell{lay_id} = WeightCell{lay_id} - ...
            reshape(Newton(segs:(segs+shift-1)),...
            [n_neuron(lay_id)+1, n_neuron(lay_id+1)]);
        segs = segs + shift;
    end
    
    WeitCellList{iterate+1} = WeightCell;
    % Stop criterion in error function value
    cost = f_error(WeightCell, X_input, Y_output);
    if cost < 1e-15
        break
    end
end

% (4) plot convergence curve
n_itr = size(WeitCellList,2);
fvalist = [];
for itr_id = 1:(n_itr-1)
    fval = 0;
    for lay_id = 1:n_layer
        fval = fval + norm( WeitCellList{itr_id}{lay_id} -...
            WeitCellList{n_itr}{lay_id}, 'fro' )^2;
    end
    fvalist = [fvalist fval];
end

size(fvalist)
figure(1)
semilogy(fvalist,'*-')
xlabel('iteration (k)')
ylabel('log ||W_k - W^*||_F^2')

% Function: evaluate the overall cost with least squares
function [f_val] = f_error(WaitCell, Xin, You)
f_val = 0;
n_laye = size(WaitCell,2);
n_sample = size(Xin,2);
%
for sam_id = 1:n_sample
    phin = [Xin(:,sam_id); 1];                   % phi input
    for lay_id = 1:(n_laye-1)
        temp = WaitCell{lay_id}'*phin;
        phin = [(0.5 * ( sqrt( temp.^2 + 1 ) - 1 ) + temp); 1];
    end
  	phin = WaitCell{n_laye}' * phin;

    f_val = f_val + (norm(You(:,sam_id)-phin,2))^2;
end
