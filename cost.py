import numpy as np
import ot

### Define infos
n_subj = 44
subjects = np.arange(n_subj) + 1

tasks = ['REST', 'S 21', 'S 51', 'S 71', 'S 22', 'S 52', 'S 72']

### Load data
prob = np.load('results/occurrence.npy')
transitions = np.load('results/transitions.npy')

### Compute cost w entropy-regulartized OT
costs = np.zeros((n_subj, prob.shape[1]-1))

### Loop over subjects
for subj in range(n_subj):
    # Extract resting prob
    rest = prob[subj,0] / prob[subj,0].sum()
    
    # Extract co-occurrence matrix at resting
    Qij = transitions[subj,0] / transitions[subj,0].sum()
    cost_matrix = -np.log(Qij)
    
    # Handle infinities
    cost_matrix[np.isinf(cost_matrix)] = 1e10
    
    ### Loop over tasks
    for i in range(1,prob.shape[1]):
        # Extract task probability
        task = prob[subj,i] / prob[subj,i].sum()
                
        # Compute OT problem
        Pij = ot.optim.cg(rest, task, cost_matrix, 1, f, df, verbose=True)
        
        # Compute cost
        cost = KL(Pij, Qij)
        print(cost)
        
        costs[subj,i-1] = cost
        
### Store results
np.save('results/cost.npy', costs)