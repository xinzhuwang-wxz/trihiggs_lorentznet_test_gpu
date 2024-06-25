import uproot
import numpy as np

def readRoot(path, treename):
    file = uproot.open(path)
    tree = file[treename]
    #Get data->numpy
    data = tree.arrays(library="np")

    jets_pt = data["jets_pt"]
    jets_eta = data["jets_eta"]
    jets_phi = data["jets_phi"]
    jets_E = data["jets_E"]

    n_jets = data["njets"]
    n_b_jets = data["nbjets"]
    total_jets_energy = data["TotalJetsEnergy"]
    total_higgs_jet_pt = data["TotalHiggsJetPt"]

    is_signal = data["isSignal"]

    return jets_E, jets_pt, jets_eta, jets_phi, n_jets, n_b_jets, total_jets_energy, total_higgs_jet_pt, is_signal

if __name__=='__main__':
    path = "data/raw_data/triHiggs_ML_test_v4.root"
    treename = "HHHNtuple"
    a = readRoot(path, treename)
    print(a)
 