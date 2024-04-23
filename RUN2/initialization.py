# % This function initialize the first population of search agents
# function X=initialization(nP,dim,ub,lb)
#
# Boundary_no= size(ub,2); % numnber of boundaries
#
# % If the boundaries of all variables are equal and user enter a signle
# % number for both ub and lb
# if Boundary_no==1
#     X=rand(nP,dim).*(ub-lb)+lb;
# end
#
# % If each variable has a different lb and ub
# if Boundary_no>1
#     for i=1:dim
#         ub_i=ub(i);
#         lb_i=lb(i);
#         X(:,i)=rand(nP,1).*(ub_i-lb_i)+lb_i;
#     end
# end
import numpy as np
def initialization(n, dim, ub, lb):
    if type(ub) == int:     # 待修改
        X = np.random.rand(n, dim) * (ub-lb) + lb
    else:
        X = np.random.rand(n, dim)
        for i in range(dim):
            X[:, i] = X[:, i] * (ub[i]-lb[i]) + lb[i]
    return X











