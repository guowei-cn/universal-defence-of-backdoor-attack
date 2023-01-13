import time

import h5py
import random

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
from utils import PCA_analysis
import os

debugging_flag = True

def draw_reachability_plot(clust, normalized_feature, labels, save_name, ground_or_pred='G', tpr=None, tnr=None):
    ordered_labels = labels[clust.ordering_]
    space = np.arange(len(normalized_feature))
    reachability = clust.reachability_[clust.ordering_]
    assert ground_or_pred == 'G' or ground_or_pred == 'P', 'Error: ground_or_pred parameter is set wrongly'
    if ground_or_pred == 'G':
        plt.scatter(space[ordered_labels == 0], reachability[ordered_labels == 0], color='yellow', marker='.',
                    alpha=0.2, label='benign_num_{}'.format(len(space[ordered_labels == 0])))
        plt.scatter(space[ordered_labels == 1], reachability[ordered_labels == 1], color='red', marker='.',
                    alpha=0.2, label='poisoned_num_{}'.format(len(space[ordered_labels == 1])))
        plt.title("RP with ground-truth label")
        plt.legend()
    elif ground_or_pred == 'P':
        plt.scatter(space[ordered_labels == 0], reachability[ordered_labels == 0], color='green', marker='.',
                    alpha=0.2, label='pre_benign_{}'.format(len(space[ordered_labels == 0])))
        plt.scatter(space[ordered_labels == 1], reachability[ordered_labels == 1], color='blue', marker='.',
                    alpha=0.2, label='pre_poisoned_{}'.format(len(space[ordered_labels == 1])))
        plt.title("RP with prediction label with tpr {:.3f} and tnr {:.3f}".format(tpr, tnr))
        plt.legend()

    plt.savefig(save_name+'.jpg')
    # plt.show()
    plt.close()


def draw_space_plot(normalized_feature, labels, save_name, ground_or_pred='G', tpr=None, fpr=None, dim=2):
    assert ground_or_pred == 'G' or ground_or_pred == 'P', 'Error: ground_or_pred parameter is set wrongly'
    assert dim == 2 or dim == 3, 'Error: dim parameter is set wrongly'
    if dim == 2:
        # draw pictures
        benign_x, benign_y, benign_label = [], [], []
        poisoned_x, poisoned_y, poisoned_labe = [], [], []
        for id, (point, label) in enumerate(zip(normalized_feature, labels)):
            if label == 1:
                poisoned_x.append(point[0])
                poisoned_y.append(point[1])
                poisoned_labe.append(label)
            else:
                benign_x.append(point[0])
                benign_y.append(point[1])
                benign_label.append(label)
        if ground_or_pred == 'G':
            plt.scatter(poisoned_x, poisoned_y, color='red', label='poisoned_num_{}'.format(len(poisoned_x)), alpha=0.3)
            plt.scatter(benign_x, benign_y, color='yellow', label='benign_num_{}'.format(len(benign_x)), alpha=0.3)
            plt.title("SP with ground-truth label")
        elif ground_or_pred == 'P':
            plt.scatter(poisoned_x, poisoned_y, color='blue', label='pre_poisoned_num_{}'.format(len(poisoned_x)), alpha=0.3)
            plt.scatter(benign_x, benign_y, color='green', label='pre_benign_num_{}'.format(len(benign_x)), alpha=0.3)
            plt.title("SP with prediction label with tpr {:.3f} and fpr {:.3f}".format(tpr, fpr))

        plt.legend()
        plt.savefig(save_name+'.jpg')
        # plt.show()
        plt.close()
    elif dim == 3:
        # draw pictures
        ax = plt.axes(projection="3d")
        benign_x, benign_y, benign_z, benign_label = [], [], [], []
        poisoned_x, poisoned_y, poisoned_z, poisoned_label = [], [], [], []
        for id, (point, label) in enumerate(zip(normalized_feature, labels)):
            if label == 1:
                poisoned_x.append(point[0])
                poisoned_y.append(point[1])
                poisoned_z.append(point[2])
                poisoned_label.append(label)
            else:
                benign_x.append(point[0])
                benign_y.append(point[1])
                benign_z.append(point[2])
                benign_label.append(label)
        if ground_or_pred == 'G':
            ax.scatter(poisoned_x, poisoned_y, poisoned_z, color='red', label='poisoned_num_{}'.format(len(poisoned_x)),
                       alpha=0.3)
            ax.scatter(benign_x, benign_y, benign_z, color='yellow', label='benign_num_{}'.format(len(benign_x)), alpha=0.3)
            plt.title("SP with ground-truth label")
        elif ground_or_pred == 'P':
            ax.scatter(poisoned_x, poisoned_y, poisoned_z, color='blue',
                       label='pre_poisoned_num_{}'.format(len(poisoned_x)),
                       alpha=0.3)
            ax.scatter(benign_x, benign_y, benign_z, color='green', label='pre_benign_num_{}'.format(len(benign_x)),
                       alpha=0.3)
            plt.title("SP with prediction label with tpr {:.3f} and tnr {:.3f}".format(tpr, fpr))
        plt.legend()
        plt.savefig(save_name+'.jpg')
        # plt.show()
        plt.close()


def draw_tpr_tnr_l(k_l, tpr_l, tnr_l, save_name):
    plt.plot(k_l, tpr_l, color='green', label='tpr')
    plt.plot(k_l, tnr_l, color='red', label='tnr')
    plt.title('k vs tpr and tnr')
    plt.xlabel('k')
    plt.legend()
    plt.savefig(save_name)


def save_poisoned_indics(gt_indics, predition, save_name):
    predicted_poisoned_indics = gt_indics[predition==1]
    np.save(save_name+'.npy', predicted_poisoned_indics)


def draw_sdas_suas(clust, sdas, suas, save_name):
    reachability = clust.reachability_[clust.ordering_]
    space = np.arange(len(reachability))
    # get label based on the sdas, suas
    # 0: not steep area
    # 1: steap down areas
    # 2: steep up areas
    labels = np.zeros(len(reachability))
    for D_area in sdas:
        labels[D_area['start']:D_area['end']+1] = 1

    for U_area in suas:
        labels[U_area['start']:U_area['end']+1] = 2


    plt.scatter(space[labels==0], reachability[labels==0], color='grey', marker='.', alpha=1, label='non steep area')
    plt.scatter(space[labels == 1], reachability[labels == 1], color='red', marker='.', alpha=1, label='steep down')
    plt.scatter(space[labels == 2], reachability[labels == 2], color='yellow', marker='.', alpha=1, label='steep up')
    plt.title("steep areas")
    plt.legend()
    # plt.show()
    plt.savefig(save_name+'.jpg', dpi=600)
    plt.close()


def plot_figures(normalized_feature, ground_truth, predition, tpr, fpr, save_path, save_name, validation_flag):
    if validation_flag:
        sub_name = 'validation'
    else:
        sub_name = 'evaluation'

    k = 2
    draw_space_plot(normalized_feature, ground_truth,
                    save_name=os.path.join(save_path + r'\{}\space_plot\ground_truth'.format(sub_name), save_name),
                    ground_or_pred='G', dim=k)
    draw_space_plot(normalized_feature, predition,
                    save_name=os.path.join(save_path + r'\{}\space_plot\prediction'.format(sub_name), save_name),
                    ground_or_pred='P', tpr=tpr, fpr=fpr, dim=k)


def distance_comparison(features: np.array, codebook: np.array)->np.array:
    distance_to_center_0 = np.sqrt(np.sum(np.square(features - codebook[0]), axis=1))
    distance_to_center_1 = np.sqrt(np.sum(np.square(features - codebook[1]), axis=1))
    prediction = (distance_to_center_0>distance_to_center_1)

    output = np.zeros_like(prediction, dtype=float)
    # new test
    for class_id in set(prediction):
        output[prediction == class_id] = np.sum(prediction == class_id) / prediction.shape[0]

    return output


def pca_plus_kmean_estimation_eval(hdf5_file, model_file, target_class):
    with h5py.File(hdf5_file, "r") as f:
        data = f['{}'.format(target_class)][:, :-1]
        indics = f['{}'.format(target_class)][:, -1]
        ground_truth = indics < 0

        center_feature_matrix = data - np.mean(data, axis=0)  # centerize the data
        score = []
        eignvs, _ = PCA_analysis(center_feature_matrix)  # the final column is the indics
        k = 2
        for i in range(k):
            score.append(np.matmul(center_feature_matrix, np.transpose(eignvs[i])))
        reduced_data = np.stack(score, axis=1)

        codebook_p, distortion_p = kmeans(reduced_data, 2)
        pred = distance_comparison(reduced_data, codebook_p)

    return pred, ground_truth


def time_calculate_AC(data):
    cla_time = 0.1
    clu_t_l = []
    for i in range(10):
        clu_start = time.time()
        center_feature_matrix = data - np.mean(data, axis=0)  # centerize the data
        score = []
        eignvs, _ = PCA_analysis(center_feature_matrix)  # the final column is the indics
        k = 2
        for i in range(k):
            score.append(np.matmul(center_feature_matrix, np.transpose(eignvs[i])))
        reduced_data = np.stack(score, axis=1)

        codebook_p, distortion_p = kmeans(reduced_data, 2)

        clu_end = time.time()

        clu_t_l.append(clu_end - clu_start)

    return np.mean(clu_t_l), cla_time

if __name__ == '__main__':
    print('corrupted label with gu trigger')
    feature_path = '../feature_and_model/corrupted_label_gu_trigger/feature_corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = '../feature_and_model/corrupted_label_gu_trigger/corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0
    class_num = 10
    pred, ground_truth = pca_plus_kmean_estimation_eval(feature_path,
                                                        model_path,
                                                        target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
    print('corrupted label with ramp trigger')
    feature_path = '../feature_and_model/corrupted_label_ramp_trigger/feature_corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.hdf5'
    model_path = '../feature_and_model/corrupted_label_ramp_trigger/corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.pt'
    target_class = 0
    pred, ground_truth = pca_plus_kmean_estimation_eval(feature_path,
                                                        model_path,
                                                        target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
    print('clean label with gu trigger')
    feature_path = '../feature_and_model/clean_label_gu_trigger/feature_clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.hdf5'
    model_path = '../feature_and_model/clean_label_gu_trigger/clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.pt'
    target_class = 2
    pred, ground_truth = pca_plus_kmean_estimation_eval(feature_path,
                                                        model_path,
                                                        target_class)
    fpr_l, tpr_l, _ = roc_curve(ground_truth, pred,
                                           pos_label=1)
    roc_auc = auc(fpr_l, tpr_l)
    print(roc_auc)
