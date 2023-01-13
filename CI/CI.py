import numpy as np
import h5py
from scipy.special import rel_entr
from sklearn import mixture
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch
import random
from PIL import ImageFilter
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from mydatasets import poisoned_MNIST, corrupted_indices_num_60000
from mymodel import Net, poisoning


mytransform = transforms.Compose([
    # transforms.RandomCrop(28, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
de_mytransform = transforms.Compose(
    [
        transforms.Normalize(mean=(-0.1307 / 0.3081,), std=(1.0 / 0.3081,)),
        transforms.ToPILImage()
    ]
)
re_mytransform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)


def get_image_data_for_cluster(best_cluster, indics, trigger_name, target_class, poison_ratio):
    # get the indics for each cluster
    indics_to_cluster_dict, indics_for_cluster, data_for_cluster = {}, {}, {}
    for cluster_id in set(best_cluster):
        data_for_cluster[cluster_id] = []
        indics_for_cluster[cluster_id] = []

    for id, cluster_id in zip(indics, best_cluster):
        indics_to_cluster_dict[int(id)] = cluster_id
    # load corresponding dataset
    worker_num = 0
    train_batch = 512
    corrupted_ds = poisoned_MNIST('data', train=True, validation=False, download=True, transform=mytransform)
    corrupted_dl = DataLoader(corrupted_ds, shuffle=True, batch_size=train_batch, num_workers=worker_num)
    # num_target = torch.sum(corrupted_ds.targets == target_class)
    # poison_num = int(num_target * poison_ratio / (1 - poison_ratio))
    # poison_index = corrupted_indices_num_60000[:poison_num]
    # assert poison_num == np.sum(indics<0), 'In get_image_data_for cluster, the poison number is not equal to original one'
    poison_index = -1 * indics[indics<0]
    for batch_idx, (data, target, index) in enumerate(corrupted_dl):
        p_data, p_target, p_index = poisoning(data, target, index, poison_index, trigger_name, target_class,
                                              de_mytransform, re_mytransform)

        if p_data != []:
            data = torch.cat([data, torch.stack(p_data)])
            target = torch.cat([target, torch.stack(p_target)])
            index = torch.cat([index, torch.stack(p_index)])

        # filter out only the target class
        data, target, index = data[target == target_class], target[target == target_class], index[
            target == target_class]

        # regroup the data accorrding to the cluster
        for img, img_i in zip(data, index):
            # average filter on image space
            img = de_mytransform(img)
            img = img.filter(ImageFilter.BoxBlur(2))  # radius = 2, so that the kernel size is 5
            img = re_mytransform(img)
            # grouping according to cluster
            cluster_id = indics_to_cluster_dict[int(img_i.cpu().item())]
            indics_for_cluster[cluster_id].append(int(img_i.cpu().item()))
            data_for_cluster[cluster_id].append(img)

    return data_for_cluster, indics_for_cluster

@torch.no_grad()
def cluster_determine(best_cluster, indics, model_file, trigger_name, target_class, poison_ratio):
    # load model
    device = torch.device("cuda")
    model = Net()
    model.load_state_dict(
        torch.load(model_file, map_location='cpu')['model']
    )
    model.to(device)
    model.eval()
    # get the image data for each cluster
    data_for_cluster, indics_for_cluster = get_image_data_for_cluster(best_cluster, indics, trigger_name, target_class, poison_ratio)

    # measure the impurity
    pr_cluster_l, index_poison, index_benign = [], [], []
    index_to_KL_dict = {}
    # for each cluster, counting the number of correct classification after average filter
    for cluster_id in set(best_cluster):
        data_one_cluster = torch.stack(data_for_cluster[cluster_id]).to(device)
        output = F.softmax(model(data_one_cluster), dim=1)
        pred = torch.argmax(output, dim=1)
        pr_cluster = torch.sum(pred == target_class) / pred.shape[0]
        pr_cluster_l.append(pr_cluster.cpu().item())
        # calculate the KL distance between [1,0] with [p, 1-p]
        KL_d = sum(rel_entr([1, 0], [pr_cluster.cpu().item(), 1 - pr_cluster.cpu().item()]))
        for i in indics_for_cluster[cluster_id]:
            index_to_KL_dict[i] = KL_d

    return index_to_KL_dict


def gmm_bic(data):
    # find the most suitable n_components
    best_bic, best_cluster, best_n_comp = np.infty, [], -1
    n_components_range = range(2, 25, 5)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type="full"
        )
        gmm.fit(data)
        bic_v = gmm.bic(data)
        if bic_v < best_bic:
            best_bic = bic_v
            best_cluster = gmm.predict(data)
            best_n_comp = n_components

    return best_cluster, best_n_comp


def gmm_plus_bic_estimation_eval(hdf5_file, model_file, target_class):
    # read the features
    poison_ratio = float(model_file.split('poison_ratio_')[1].split('.pt')[0])
    trigger_name = model_file.split('tri_name_')[1].split('_target_class')[0]

    with h5py.File(hdf5_file, "r") as f:
        data = f['{}'.format(target_class)][:, :-1]
        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0

    # get cluster with best n_component
    best_cluster, best_n_comp = gmm_bic(data)
    index_to_KL_dict = cluster_determine(best_cluster, indics, model_file, trigger_name, target_class, poison_ratio) # for each cluster check whether it is subset of poisoned data

    pred = np.zeros_like(ground_truth, dtype=float)
    for i, index in enumerate(indics):
        pred[i] = index_to_KL_dict[index]
    # get the prediction and calculate the tpr and fpr
    return pred, ground_truth, best_n_comp


def gmm_plus_bic_estimation_eval_benign(hdf5_file, model_file, target_class):
    # read the features
    poison_ratio = float(model_file.split('poison_ratio_')[1].split('.pt')[0])

    with h5py.File(hdf5_file, "r") as f:
        data = f['{}'.format(target_class)][:, :-1]
        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0

    # get cluster with best n_component
    best_cluster, best_n_comp = gmm_bic(data)
    index_to_KL_dict = cluster_determine_benign(best_cluster, indics, model_file, target_class) # for each cluster check whether it is subset of poisoned data

    pred = np.zeros_like(ground_truth, dtype=float)
    for i, index in enumerate(indics):
        pred[i] = index_to_KL_dict[index]
    # get the prediction and calculate the tpr and fpr
    return pred, ground_truth, best_n_comp


def get_image_data_for_cluster_benign(best_cluster, indics, target_class):
    # get the indics for each cluster
    indics_to_cluster_dict, indics_for_cluster, data_for_cluster = {}, {}, {}
    for cluster_id in set(best_cluster):
        data_for_cluster[cluster_id] = []
        indics_for_cluster[cluster_id] = []

    for id, cluster_id in zip(indics, best_cluster):
        indics_to_cluster_dict[int(id)] = cluster_id
    # load corresponding dataset
    worker_num = 8
    train_batch = 512
    corrupted_ds = poisoned_MNIST('data', train=True, validation=False, download=True, transform=mytransform)
    corrupted_dl = DataLoader(corrupted_ds, shuffle=True, batch_size=train_batch, num_workers=worker_num)
    for batch_idx, (data, target, index) in enumerate(corrupted_dl):
        # filter out only the target class
        data, target, index = data[target == target_class], target[target == target_class], index[target == target_class]

        # regroup the data accorrding to the cluster
        for img, img_i in zip(data, index):
            # average filter on image space
            img = de_mytransform(img)
            img = img.filter(ImageFilter.BoxBlur(2))  # average filter 3by3 and radius=2 is 3-1
            img = re_mytransform(img)
            # grouping according to cluster
            cluster_id = indics_to_cluster_dict[int(img_i.cpu().item())]
            indics_for_cluster[cluster_id].append(int(img_i.cpu().item()))
            data_for_cluster[cluster_id].append(img)

    return data_for_cluster, indics_for_cluster



@torch.no_grad()
def cluster_determine_benign(best_cluster, indics, model_file, target_class):
    # load model
    device = torch.device("cuda:2")
    model = Net()
    model.load_state_dict(
        torch.load(model_file, map_location='cpu')['model']
    )
    model.to(device)
    model.eval()
    # get the image data for each cluster
    data_for_cluster, indics_for_cluster = get_image_data_for_cluster_benign(best_cluster, indics, target_class)

    # measure the impurity
    pr_cluster_l, index_poison, index_benign = [], [], []
    index_to_KL_dict = {}
    # for each cluster, counting the number of correct classification after average filter
    for cluster_id in set(best_cluster):
        data_one_cluster = torch.stack(data_for_cluster[cluster_id]).to(device)
        output = F.softmax(model(data_one_cluster), dim=1)
        pred = torch.argmax(output, dim=1)
        pr_cluster = torch.sum(pred == target_class) / pred.shape[0]
        pr_cluster_l.append(pr_cluster.cpu().item())
        # calculate the KL distance between [1,0] with [p, 1-p]
        KL_d = sum(rel_entr([1, 0], [pr_cluster.cpu().item(), 1 - pr_cluster.cpu().item()]))
        for i in indics_for_cluster[cluster_id]:
            index_to_KL_dict[i] = KL_d

    return index_to_KL_dict


if __name__ == '__main__':
    print('corrupted label with gu trigger')
    feature_path = '../feature_and_model/corrupted_label_gu_trigger/feature_corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = '../feature_and_model/corrupted_label_gu_trigger/corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0
    class_num = 10
    pred, ground_truth, best_n_comp = gmm_plus_bic_estimation_eval(feature_path,
        model_path, target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
    print('corrupted label with ramp trigger')
    feature_path = '../feature_and_model/corrupted_label_ramp_trigger/feature_corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = '../feature_and_model/corrupted_label_ramp_trigger/corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0
    pred, ground_truth, best_n_comp = gmm_plus_bic_estimation_eval(feature_path,
        model_path, target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
    print('clean label with gu trigger')
    feature_path = '../feature_and_model/clean_label_gu_trigger/feature_clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.hdf5'
    model_path = '../feature_and_model/clean_label_gu_trigger/clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.pt'
    target_class = 2
    pred, ground_truth, best_n_comp = gmm_plus_bic_estimation_eval(feature_path,
        model_path, target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
