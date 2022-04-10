"""
Code taken from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/evaluate.py
"""

import argparse
import math
import numpy as np
import socket
import importlib
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.loading_pointclouds import *
import models.KPConvNetVLAD as KPN
import utils.pointnetvlad_loss as PNV_loss

import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('evaluation')
parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                    choices=['mean', 'max'])
parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    
args = parser.parse_args()

def evaluate():
    model = KPN.KPConvNetVLAD()
    model = model.to(device)

    resume_filename = cfg.RESUME_FILENAME
    print("Evaluating using pretrain model", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    print('average one percent recall', evaluate_model(model))


def evaluate_model(model):
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    try:
        DATABASE_VECTORS = np.load(os.path.join(cfg.RESULTS_FOLDER,'database_vectors.npy'), allow_pickle=True)
        QUERY_VECTORS = np.load(os.path.join(cfg.RESULTS_FOLDER, 'query_vectors.npy'), allow_pickle=True)
    except:
        # generate descriptors from input point clouds
        for i in range(len(DATABASE_SETS)):
            DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

        for j in range(len(QUERY_SETS)):
            QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

        np.save(os.path.join(cfg.RESULTS_FOLDER,'database_vectors.npy'), np.array(DATABASE_VECTORS))
        np.save(os.path.join(cfg.RESULTS_FOLDER, 'query_vectors.npy'), np.array(QUERY_VECTORS))

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    plot_average_recall_curve(ave_recall)

    # precision-recall curve
    get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, ave_one_percent_recall)
    
    return ave_one_percent_recall


def plot_average_recall_curve(ave_recall):
    index = np.arange(1, 26)
    plt.figure()
    plt.plot(index, ave_recall, label='KPConv-NetVLAD')

    try:
        ave_recall_pointnetvlad = ''
        with open(os.path.join(cfg.POINTNETVLAD_RESULTS_FOLDER, 'results.txt'), "r") as pointnetvlad_result_file:
            ave_recall_pointnetvlad_temp = pointnetvlad_result_file.readlines()[1:6]
            for i in range(len(ave_recall_pointnetvlad_temp)):
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace('[', '')
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace(']', '')
                ave_recall_pointnetvlad_temp[i] = ave_recall_pointnetvlad_temp[i].replace('\n', '')
                ave_recall_pointnetvlad = ave_recall_pointnetvlad + ave_recall_pointnetvlad_temp[i]
            ave_recall_pointnetvlad = np.array(ave_recall_pointnetvlad.split())
            ave_recall_pointnetvlad = np.asarray(ave_recall_pointnetvlad, dtype = float)
        plt.plot(index, ave_recall_pointnetvlad, 'k--', label='PointNetVLAD')
    except:
        print('no pointnetvlad')
    
    plt.title("Average recall @N Curve")
    plt.xlabel('in top N')
    plt.ylabel('Average recall @N [%]')
    # plt.xlim(-1,26)
    # plt.ylim(0,105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_FOLDER, "average_recall_curve.png"))
    print('Average recall curve is saved at:', os.path.join(cfg.RESULTS_FOLDER, "average_recall_curve.png"))



def get_precision_recall_curve(QUERY_SETS, QUERY_VECTORS, DATABASE_VECTORS, ave_one_percent_recall):
    y_true = []
    y_predicted = []

    for q in range(len(QUERY_SETS)):
        for d in range(len(QUERY_SETS)):
            if (q==d):
                continue

            database_nbrs = KDTree(DATABASE_VECTORS[d])

            for i in range(len(QUERY_SETS[q])):
                true_neighbors = QUERY_SETS[q][i][d]
                if(len(true_neighbors)==0):
                    continue
                distances, indices = database_nbrs.query(np.array([QUERY_VECTORS[q][i]]))
                current_y_true = 0
                current_y_predicted = 0
                for j in range(len(indices[0])):
                    if indices[0][j] in true_neighbors:
                        # predicted neighbor is correct
                        current_y_true = 1
                    current_y_predicted_temp = np.dot(QUERY_VECTORS[q][i], DATABASE_VECTORS[d][indices[0][j]]) / \
                                                    (np.linalg.norm(QUERY_VECTORS[q][i]) * np.linalg.norm(DATABASE_VECTORS[d][indices[0][j]]))
                    # take prediction similarity that is the highest amoung neighbors
                    if current_y_predicted_temp > current_y_predicted:
                        current_y_predicted = current_y_predicted_temp
                # loop or not
                y_true.append(current_y_true)

                # similarity
                y_predicted.append(current_y_predicted)
    
    np.set_printoptions(threshold=sys.maxsize)
    # print('y_true', y_true)
    # print('y_predicted', y_predicted)

    precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)

    # print('precision', precision)
    # print('recall', recall)
    # print('thresholds', thresholds)
    np.set_printoptions(threshold=1000)

    np.save(os.path.join(cfg.RESULTS_FOLDER, 'precision.npy'), np.array(precision))
    np.save(os.path.join(cfg.RESULTS_FOLDER, 'recall.npy'), np.array(recall))

    # Plot Precision-recall curve
    plt.figure()
    plt.plot(recall*100, precision*100, label='KPConv-NetVLAD, average recall=%.2f' % (ave_one_percent_recall))
    try:
        ave_one_percent_recall_pointnetvlad = None
        with open(os.path.join(cfg.POINTNETVLAD_RESULTS_FOLDER, 'results.txt'), "r") as pointnetvlad_result_file:
            ave_one_percent_recall_pointnetvlad = float(pointnetvlad_result_file.readlines()[-1])
        precision_pointnetvlad = np.load(os.path.join(cfg.POINTNETVLAD_RESULTS_FOLDER, 'precision.npy'))
        recall_pointnetvlad = np.load(os.path.join(cfg.POINTNETVLAD_RESULTS_FOLDER, 'recall.npy'))
        plt.plot(recall_pointnetvlad*100, precision_pointnetvlad*100, 'k--', label='PointNetVLAD, average recall=%.2f' % (ave_one_percent_recall_pointnetvlad))
    except:
        print('no baseline')
        

    plt.title("Precision-recall Curve")
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.xlim(0,105)
    plt.ylim(0,105)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_FOLDER, "precision_recall_oxford.png"))
    print('Precision-recall curve is saved at:', os.path.join(cfg.RESULTS_FOLDER, "precision_recall_oxford.png"))


def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    eval_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(eval_file_idxs)//batch_num):
        file_indices = eval_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out, _ = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(eval_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = eval_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1, _ = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    # model.train()
    # print(q_output.shape)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    evaluate()