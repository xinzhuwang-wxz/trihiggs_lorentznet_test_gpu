import torch
import numpy as np
from torch.utils.data import Dataset
from read_root import readRoot
'''
{'branch':value}
'''

class makeDataset(Dataset):
    def __init__(self, root_file, treename, shuffle=True):
        super().__init__()
        jets_E, jets_pt, jets_eta, jets_phi, n_jets, n_b_jets, total_jets_energy, total_higgs_jet_pt, is_signal = readRoot(root_file, treename)
        self.num_events = len(is_signal)
        
        self.data = {
            'jets_p4': [], 
            'jets_E': jets_E, 
            'jets_pt': jets_pt, 
            'jets_eta': jets_eta,
            'jets_phi': jets_phi, 
            'n_jets': n_jets,
            'n_b_jets': n_b_jets,
            'total_jets_energy': total_jets_energy,
            'total_higgs_jet_pt': total_higgs_jet_pt,
            'is_signal': is_signal,
        }
        
        
        for i in range(self.num_events):
            px = jets_pt[i] * np.cos(jets_phi[i])
            py = jets_pt[i] * np.sin(jets_phi[i])
            pz = jets_pt[i] * np.sinh(jets_eta[i])
            p4 = np.vstack((jets_E[i], px, py, pz)).T
            self.data['jets_p4'].append(p4)
        

        if shuffle:
            self.perm = torch.randperm(self.num_events)
        else:
            self.perm = torch.arange(self.num_events)


    def __len__(self): # a = JetDataset(data), len(a)即可调用
        return self.num_events

    def __getitem__(self, idx):  # a = JetDataset(data), a[2]即可调用
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
    
if __name__ == "__main__":
    root_file = "data/raw_data/triHiggs_ML_test_v4.root"  
    treename = "HHHNtuple" 
    dataset = makeDataset(root_file, treename)
    

    for i in range(2):
        print(f"Event {i+1}:")
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
            print(type(value))
        print()


    
    