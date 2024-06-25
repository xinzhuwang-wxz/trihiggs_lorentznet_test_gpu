import torch
import numpy as np
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
from init_dataset import init_dataset

enc = OneHotEncoder().fit([[-1], [1]])

def batch_stack(props, edge_mat=False, nobj=None):
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.tensor(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

def drop_zeros(props, to_keep):
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0 or props[0].shape[0] != to_keep.shape[0]:
        return props
    else:
        return props[:, to_keep, ...]

def normsq4(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)

def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx * n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)], dim=0)
    return edges

def collate_fn(data, scale=1., add_beams=False, beam_mass=1):


    # 批量堆叠数据
    data = {key: batch_stack([item[key] for item in data]) for key in data[0].keys()}

    if add_beams:
        beams = torch.tensor([[[sqrt(1 + beam_mass**2), 0, 0, 1], [sqrt(1 + beam_mass**2), 0, 0, -1]]], dtype=torch.float32).expand(data['jets_p4'].shape[0], 2, 4)
        s = data['jets_p4'].shape
        data['jets_p4'] = torch.cat([beams * scale, data['jets_p4'] * scale], dim=1)
        labels = torch.cat((torch.ones(s[0], 2), -torch.ones(s[0], s[1])), dim=1)
        if 'scalars' not in data.keys():
            data['scalars'] = labels.to(dtype=data['jets_p4'].dtype)#.unsqueeze(-1)
        else:
            data['scalars'] = torch.cat((data['scalars'], labels.to(dtype=data['jets_p4'].dtype)), dim=1)
    else:
        data['jets_p4'] = data['jets_p4'] * scale

    atom_mask = data['jets_p4'][..., 0] != 0.
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    data['atom_mask'] = atom_mask.to(torch.bool)
    data['edge_mask'] = edge_mask.to(torch.bool)

    batch_size, n_nodes, _ = data['jets_p4'].size()

    # 合并标量数据到节点属性中
    scalar_values = torch.cat([data[key].unsqueeze(1) for key in ['n_jets', 'n_b_jets', 'total_jets_energy', 'total_higgs_jet_pt']], dim=1)
    scalar_values = scalar_values.unsqueeze(1).expand(-1, n_nodes, -1)
    nodes = torch.cat([data['jets_p4'], scalar_values], dim=-1)

    if add_beams:
        beamlabel = data['scalars']
        one_hot = enc.transform(beamlabel.reshape(-1, 1)).toarray().reshape(batch_size, n_nodes, -1)
        one_hot = torch.tensor(one_hot)
        mass = normsq4(data['jets_p4']).abs().sqrt().unsqueeze(-1)
        mass_tensor = mass.view(mass.shape + (1,))
        nodes = torch.cat([nodes, (one_hot.unsqueeze(-1) * mass_tensor).view(mass.shape[:2] + (-1,))], dim=-1)
    else:
        mass = normsq4(data['jets_p4']).unsqueeze(-1)
        nodes = torch.cat([nodes, mass], dim=-1)

    edges = get_adj_matrix(n_nodes, batch_size, data['edge_mask'])
    data['nodes'] = nodes
    data['edges'] = edges

    return data # 就得这么返回诶

# 示例代码，用于测试 collate_fn
if __name__ == "__main__":
    datadir = "data/raw_data"
    datasets = init_dataset(datadir)
    
    train_dataset = datasets['train']
    batch = [train_dataset[i] for i in range(2)]  # 取2个样本进行测试
    data = collate_fn(batch, add_beams=True)

    for key, val in data.items():
        print(f"Data key: {key}, Data value shape: {val.shape}")



'''
Data key: jets_p4, Data value shape: torch.Size([2, 8, 4])
Data key: jets_E, Data value shape: torch.Size([2, 6])
Data key: jets_pt, Data value shape: torch.Size([2, 6])
Data key: jets_eta, Data value shape: torch.Size([2, 6])
Data key: jets_phi, Data value shape: torch.Size([2, 6])
Data key: n_jets, Data value shape: torch.Size([2])
Data key: n_b_jets, Data value shape: torch.Size([2])
Data key: total_jets_energy, Data value shape: torch.Size([2])
Data key: total_higgs_jet_pt, Data value shape: torch.Size([2])
Data key: is_signal, Data value shape: torch.Size([2])
Data key: scalars, Data value shape: torch.Size([2, 8])
Data key: atom_mask, Data value shape: torch.Size([2, 8])
Data key: edge_mask, Data value shape: torch.Size([2, 8, 8])
Data key: nodes, Data value shape: torch.Size([2, 8, 10])
Data key: edges, Data value shape: torch.Size([2, 112])根据这个输出，为网络弄一个测试代码
'''