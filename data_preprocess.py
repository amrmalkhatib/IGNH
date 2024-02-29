from torch.utils.data import Dataset
from scipy.stats import pearsonr, pointbiserialr, kendalltau
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Data(Dataset):

    def __init__(self, data):
        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = data

    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        example, label = self._samples[index]

        return example, label

    def __len__(self):
        """Total number of samples"""
        return len(self._samples)

    def get_sample_by_id(self, id_):
        id_index = [path.stem for (path, _) in self._samples].index(id_)
        return self[id_index]


class BlackBoxWrapper():
    def __init__(self, model,
                 num_players,
                 device):
        self.model = model
        self.num_players = num_players
        #self.scaler = scaler
        self.device = device

    def __call__(self, x, S):
        '''
        Evaluate a black-box model.
        Args:
          x: input examples.
          S: coalitions.
        '''
        #x = self.scaler.transform(x)
        x = x * S

        #x = self.scaler.inverse_transform(x)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.reshape((x.shape[0], self.num_players, 1))
        values = self.model(x)

        return values


def transform_to_tensors(data, labels, adj_matrix):
    data_array = data.values  # Convert DataFrame to NumPy array
    tensor_data = torch.FloatTensor(data_array).view(-1, adj_matrix.shape[0], 1)
    print(tensor_data.shape)

    tensor_labels = torch.tensor(labels.values, dtype=torch.float)#.view(-1, 1)
    print(tensor_labels.shape)

    list_of_tensors = list(zip(tensor_data, tensor_labels))
    return list_of_tensors


def compute_adjacency_matrix(data, self_loop_weight, categorical, p_value_thr=0.05):
    index_to_name = {i: n for i, n in enumerate(data.columns)}
    name_to_index = {n: i for i, n in enumerate(data.columns)}

    adj_matrix = np.zeros((len(data.columns), len(data.columns)), dtype=float)

    for i in tqdm(range(len(data.columns))):
        for j in range(i, len(data.columns)):
            col_1 = data.columns[i]
            col_2 = data.columns[j]
            corr = np.nan
            if col_1 == col_2:
                adj_matrix[name_to_index[col_1], name_to_index[col_2]] = self_loop_weight
            else:
                if col_1 in categorical and col_2 in categorical:
                    corr, p_value = kendalltau(data[col_1], data[col_2])
                    corr = np.linalg.norm(corr)
                elif col_1 in categorical:
                    encoded_data = pd.get_dummies(data[col_1], columns=categorical)
                    bis_corr = []
                    for col in encoded_data.columns:
                        corr, p_value = pointbiserialr(data[col_1], encoded_data[col])
                        if p_value < p_value_thr:
                            bis_corr.append(corr)
                    fisher_z_values = [np.arctanh(r) for r in bis_corr]
                    z = np.mean(fisher_z_values)
                    corr = np.tanh(z)
                    p_value = 0

                elif col_2 in categorical:
                    encoded_data = pd.get_dummies(data[col_2], columns=categorical)
                    bis_corr = []
                    for col in encoded_data.columns:
                        corr, p_value = pointbiserialr(data[col_2], encoded_data[col])
                        if p_value < p_value_thr:
                            bis_corr.append(corr)
                    fisher_z_values = [np.arctanh(r) for r in bis_corr]
                    z = np.mean(fisher_z_values)
                    corr = np.tanh(z)
                    p_value = 0

                else:
                    corr, p_value = pearsonr(data[col_1], data[col_2])

                if np.isnan(corr) or p_value > p_value_thr:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = 0
                    adj_matrix[name_to_index[col_2], name_to_index[col_1]] = 0
                else:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = corr
                    adj_matrix[name_to_index[col_2], name_to_index[col_1]] = corr

    return adj_matrix, index_to_name, name_to_index


def min_max_normalize(input_data, training_data):
    normalized_data = (input_data - training_data.min()) / (training_data.max() - training_data.min())
    return normalized_data


def plot_roc(clf, X, Y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_estimator(clf, X, Y)
    plt.show()
