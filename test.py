import glob
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import pandas as pd
from tqdm import tqdm
from DEFORM_func import DEFORM_func
from DEFORM_sim import DEFORM_sim
from util import computeLengths, computeEdges, compute_u0, parallelTransportFrame
import pickle
import random
import torch.nn as nn

random.seed(0)
torch.manual_seed(0)


def resample_nodes(current_nodes, num_nodes):
    # Resample the number of nodes in the history data
    segments = []
    for idx in range(1, len(current_nodes)):
        distlist = current_nodes[idx] - current_nodes[idx - 1]
        dist = [i**2 for i in distlist]
        segments.append(np.sqrt(sum(dist)))

    total_length = sum(segments)
    indiv_length = total_length / (num_nodes - 1)

    resampled_nodes = []
    for idx in range(num_nodes):
        length = idx * indiv_length
        for i in range(len(segments)):
            if length < segments[i]:
                break
            length -= segments[i]

        ratio = length / segments[i]
        resampled_nodes.append(
            current_nodes[i] + ratio * (current_nodes[i + 1] - current_nodes[i])
        )

    return resampled_nodes



def test(DEFORM_func, DEFORM_sim, device):
    state_dict = torch.load('/home/cao/DEFORM/save_model/DLO1_15320.pth')
    n_vert = 13
    DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
    DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert-1, pbd_iter=10, device=device)
    DEFORM_sim.load_state_dict(state_dict)
    DEFORM_sim.eval()  # Set the model to evaluation mode

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float32)

    data = np.load('/home/cao/DEFORM/data_set/nooclusion_0902.npy', allow_pickle=True)

    history_positions = np.array([
        np.array(resample_nodes(node_pos, n_vert))
        for node_pos in data
    ])
    for i in range(len(history_positions)):
        history_positions[i] = np.matmul(history_positions[i], rotation_matrix) * 4 -1

    history_positions = torch.from_numpy(history_positions)
    history_positions = history_positions.unsqueeze(0).float().to(device)

    eval_previous_vertices, eval_vertices, eval_target_vertices = history_positions, history_positions[:, 1:, :, :], history_positions[:, 2:, :, :]


    with torch.no_grad():
        eval_time = 0
        eval_batch = 1
        clamped_index = torch.zeros(n_vert)
        clamped_selection = torch.tensor((0, 1, -2, -1))
        clamped_index[clamped_selection] = torch.tensor((1.))

        dir_path = "/home/cao/DEFORM/test_results"
        init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).to(device).unsqueeze(dim=0)
        inputs = eval_target_vertices[:, :, clamped_selection]
        """
        initialize all theta = 0
        """

        theta_full = torch.zeros(eval_batch, n_vert - 1).to(device)
        for traj_num in range(eval_target_vertices.size()[1]):
            with torch.no_grad():
                if traj_num == 0:
                    rest_edges = computeEdges(eval_vertices[:, traj_num])

                    m_u0 = DEFORM_func.compute_u0(rest_edges[:, 0].float(), init_direction.repeat(eval_batch, 1, 1)[:, 0])
                    current_v = (eval_vertices[:, traj_num] - eval_previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                    m_restEdgeL = DEFORM_sim.m_restEdgeL.repeat(eval_batch, 1)
                    DEFORM_sim.m_restWprev, DEFORM_sim.m_restWnext, DEFORM_sim.learned_pmass = DEFORM_sim.Rod_Init(eval_batch, init_direction.repeat(eval_batch, 1, 1), m_restEdgeL, clamped_index)
                    init_pred_vert_0, current_v, theta_full = DEFORM_sim(eval_vertices[:, traj_num], current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")

                    """visualization: store image into local file for visualization"""
                    init_vis_vert = torch.Tensor.numpy(init_pred_vert_0.to('cpu'))
                    vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                    ax.plot(init_vis_vert[0, :, 0], init_vis_vert[0, :, 1], init_vis_vert[0, :, 2], label='pred')
                    ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='tracking results')
                    ax.set_xlim(-.5, 1.)
                    ax.set_ylim(-1, .5)
                    ax.set_zlim(0, 1.)
                    plt.legend()
                    plt.savefig(dir_path + '/%s.png' % (traj_num))

                if traj_num == 1:
                    previous_edge = computeEdges(eval_previous_vertices[:, traj_num])
                    current_edges = computeEdges(init_pred_vert_0)
                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                    pred_vert, current_v, theta_full = DEFORM_sim(init_pred_vert_0, current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")
                    vert = init_pred_vert_0.clone()

                    vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                    vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                    ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2], label='pred')
                    ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='tracking results')
                    ax.set_xlim(-.5, 1.)
                    ax.set_ylim(-1, .5)
                    ax.set_zlim(0, 1.)
                    plt.legend()
                    plt.savefig(dir_path + '/%s.png' % (traj_num))

                if traj_num >= 2:
                    previous_vert = vert.clone()
                    vert = pred_vert.clone()
                    current_v = current_v.clone()
                    m_u0 = m_u0.clone()
                    previous_edge = computeEdges(previous_vert)
                    current_edges = computeEdges(vert)
                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0],m_u0)
                    pred_vert, current_v, theta_full = DEFORM_sim(vert, current_v,init_direction.repeat(eval_batch, 1, 1),clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full, mode = "evaluation")

                    vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                    vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                    ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2],label='pred')
                    ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='tracking results')
                    ax.set_xlim(-.5, 1.)
                    ax.set_ylim(-1, .5)
                    ax.set_zlim(0, 1.)
                    plt.legend()
                    plt.savefig(dir_path + '/%s.png' % (traj_num))

            eval_time += 1


if __name__ == "__main__":
    '''
    DLO_type: DLO type name, related to training dataset folder, saved model name and loss record. For loss record, 
        try to explore using tensor board
    DLO_type: DLO1/DLO2/DLO3/DLO4/DLO5
    eval/train set number = number of pickle file
    eval/train time horizon: in this case, FPS = 100 hz. change self.dt in DEFROM_sim but test it stability first 
    batch: training batch. eval batch default = eval set number
    device: cuda:0/CPU switchable
    '''
    test(DEFORM_func=DEFORM_func, DEFORM_sim=DEFORM_sim, device="cpu")

