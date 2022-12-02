import numpy as np
from functools import partial
import pdb
import torch
import torch.nn.functional as F
import math

def cost_matrix(x, y, cost_type='cosine', **kwargs):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"

    # NOTE: cosine distance and Euclidean distance
    if cost_type == 'cosine':  # 二维的
        cos_cost = []
        for i in range(x.shape[0]):
            x_lin = torch.div(x[i], torch.norm(x[i], p=2, dim=-1, keepdim=True))
            y_col = torch.div(y[i], torch.norm(y[i], p=2, dim=-1, keepdim=True))
            try:
                tmp1 = torch.matmul(x_lin, y_col.T)
            except:
                print()
            dis = 1 - tmp1  # cosine 越小的的cost越大
            cos_cost.append(dis)
        dis_final = torch.stack(cos_cost, dim=0)
    elif cost_type == 'Euclidean':  # 二维的
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        dis_final = torch.sum((torch.abs(x_col - y_lin)) ** 2, 2)
    elif cost_type == 'OPW_distance':
        order_preserved_cost = []
        for i in range(x.shape[0]):
            n, _ = x[i].shape
            m, _ = y[i].shape
            pos_x = torch.arange(1, n+1, step=1, device=x.device).unsqueeze(1).expand(-1, m)
            pos_y = torch.arange(1, m+1, step=1, device=y.device).unsqueeze(0).expand(n, -1)
            E = torch.div(1, (pos_x-pos_y)**2+1)
            F = torch.div(torch.abs(pos_x-pos_y), torch.sqrt(torch.tensor(1/(n*n)+1/(m*m), device=x.device)))
            lambda_1 = kwargs['lambda_1']
            lambda_2 = kwargs['lambda_2']
            sigma = kwargs['sigma']
            dis = -lambda_1*E + lambda_2*(F/(2*sigma**2) + torch.log(torch.tensor(sigma*math.sqrt(2*3.14), device=x.device)))
            order_preserved_cost.append(dis)
        dis_final = torch.stack(order_preserved_cost, dim=0)
    else:
        print("name error")
        dis_final = 0
    return dis_final   # 查看一下 cost matrix 是否可以使用负值


def IPOT(C, n, m, beta=0.5):
    # sigma = tf.scalar_mul(1 / n, tf.ones([n, 1]))

    sigma = torch.ones([m, 1]) / m.to(torch.float32)
    T = torch.ones([n, m])
    A = torch.exp(-C / beta)
    for t in range(50):
        Q = torch.multiply(A, T)
        for k in range(1):
            delta = 1 / (n.to(torch.float32) * torch.matmul(Q, sigma))
            sigma = 1 / (m.to(torch.float32) * torch.matmul(Q.T, delta))
        # pdb.set_trace()
        tmp = torch.matmul(torch.diag(torch.squeeze(delta)), Q)
        T = torch.matmul(tmp, torch.diag(torch.squeeze(sigma)))
    return T


def IPOT_np(C, beta=0.5):
    n, m = C.shape[0], C.shape[1]
    sigma = np.ones([m, 1]) / m
    T = np.ones([n, m])
    A = np.exp(-C / beta)
    for t in range(20):
        Q = np.multiply(A, T)
        for k in range(1):
            delta = 1 / (n * (Q @ sigma))
            sigma = 1 / (m * (Q.T @ delta))
        # pdb.set_trace()
        tmp = np.diag(np.squeeze(delta)) @ Q
        T = tmp @ np.diag(np.squeeze(sigma))
    return T


def IPOT_distance(C, n, m):
    T = IPOT(C, n, m)
    distance = torch.trace(torch.matmul(C.T, T))
    return distance





def IPOT_alg(C, beta=1.0, t_steps=10, k_steps=1):
    b, n, m = C.shape
    device = C.device
    m_tensor = torch.tensor(m, device=device, dtype=torch.float32)
    n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
    sigma = torch.ones([b, m, 1], device=device) / m_tensor  # [b, m, 1]
    T = torch.ones([b, n, m], device=device)
    A = torch.exp(-C / beta).to(torch.float32)  # [b, n, m]
    for t in range(t_steps):
        Q = A * T  # [b, n, m]
        for k in range(k_steps):
            delta = 1 / (n_tensor *
                         torch.matmul(Q, sigma))  # [b, n, 1]
            sigma = 1 / (m_tensor * torch.matmul(Q.permute([0, 2, 1]), delta))  # [b, m, 1]
        T = delta * Q * torch.permute(sigma, [0, 2, 1])  # [b, n, m]
    #    distance = tf.trace(tf.matmul(C, T, transpose_a=True))
    return T


def IPOT_distance2(C, beta=1, t_steps=10, k_steps=1):
    b, n, m = C.shape
    sigma = torch.ones([b, m, 1]) / m.to(torch.float32)  # [b, m, 1]
    T = torch.ones([b, n, m])
    A = torch.exp(-C / beta)  # [b, n, m]
    for t in range(t_steps):
        Q = A * T  # [b, n, m]
        for k in range(k_steps):
            delta = 1 / (n.to(torch.float32) * torch.matmul(Q, sigma))  # [b, n, 1]
            sigma = 1 / (m.to(torch.float32) * torch.matmul(Q.T, delta))  # [b, m, 1]
        T = delta * Q * torch.permute(sigma, [0, 2, 1])  # [b, n, m]
    distance = torch.trace(torch.matmul(C.T, T))
    return distance


def GW_alg(Cs, Ct, beta=0.5, iteration=5, OT_iteration=5, Other_C=None):
    bs, _, n = Cs.shape
    _, _, m = Ct.shape
    device = Cs.device
    m_tensor = torch.tensor(m, device=device, dtype=torch.float32)
    n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
    one_m = torch.ones([bs, m, 1], device=device) / m_tensor
    one_n = torch.ones([bs, n, 1], device=device) / n_tensor
    p = torch.ones([bs, m, 1], device=device) / m_tensor
    q = torch.ones([bs, n, 1], device=device) / n_tensor

    Cst = torch.matmul(torch.matmul(Cs ** 2, q), torch.permute(one_m, [0, 2, 1])) + \
          torch.matmul(one_n, torch.matmul(p.permute([0, 2, 1]), (Ct ** 2).permute([0, 2, 1])))
    Cst = Cst + Other_C
    gamma = torch.matmul(q, p.permute([0, 2, 1]))

    for i in range(iteration):
        tmp1 = torch.matmul(Cs, gamma)
        C_gamma = Cst - 2 * torch.matmul(tmp1, Ct.permute([0, 2, 1]))

        gamma = IPOT_alg(C_gamma, beta=beta, t_steps=OT_iteration)
    Cgamma = Cst - 2 * torch.matmul(
        torch.matmul(Cs, gamma), torch.permute(Ct, [0, 2, 1]))
    # pdb.set_trace()
    return gamma, Cgamma


def prune(dist, beta=0.1):
    min_score = torch.amin(dist, dim=[1, 2], keepdim=True)
    max_score = torch.amax(dist, dim=[1, 2], keepdim=True)
    # pdb.set_trace()
    # min_score = dist.min()
    # max_score = dist.max()
    threshold = min_score + beta * (max_score - min_score)
    res = dist - threshold
    return F.relu(res)



def GW_distance(Cs, Ct, beta=0.5, iteration=5, OT_iteration=20):
    T, Cst = GW_alg(Cs, Ct, beta=beta, iteration=iteration, OT_iteration=OT_iteration)
    GW_distance = torch.trace(torch.matmul(Cst.T, T))
    return GW_distance


def FGW_distance(Cs, Ct, C, beta=0.5, iteration=5, OT_iteration=20):
    T, Cst = GW_alg(Cs, Ct, beta=beta, iteration=iteration,
                    OT_iteration=OT_iteration)
    GW_distance = torch.trace(torch.matmul(Cst.T, T))
    W_distance = torch.trace(torch.matmul(C.T, T))
    return GW_distance, W_distance



if __name__ == '__main__':
    """
    def cost_matrix(x, y, cost_type='cosine', **kwargs):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"

    # NOTE: cosine distance and Euclidean distance
    if cost_type == 'cosine':
        x = torch.norm(x, p=2)
        y = torch.norm(y, p=2)
        tmp1 = torch.matmul(x, y.T)
        dis = 1 - tmp1
    elif cost_type == 'Euclidean':
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        dis = torch.sum((torch.abs(x_col - y_lin)) ** 2, 2)
    elif cost_type == 'OPW_distance':
        n, _ = x.shape
        m, _ = y.shape
        pos_x = torch.arange(1, n+1, step=1, device=x.device)
        pos_y = torch.arange(1, m+1, step=1, device=y.device)
        E = torch.div(1, (pos_x-pos_y)**2+1)
        F = torch.div(torch.abs(pos_x-pos_y), torch.sqrt(1/(n*n)+1/(m*m)))
        lambda_1 = kwargs['lambda_1']
        lambda_2 = kwargs['lambda_2']
        sigma = kwargs['sigma']
        dis = -lambda_1*E + lambda_2*(F/(2*sigma**2) + torch.log(sigma*math.sqrt(2*3.14)))
    else:
        print("name error")
        dis = 0
    return dis
    """
    x = torch.randn(4, 512)
    y = torch.randn(6, 512)

    C = cost_matrix(x, y, cost_type='OPW_distance', lambda_1=1, lambda_2=0.1, sigma=1)  # 使用IPOT cost的值不能太大
    print(C)
    print(IPOT(C, torch.tensor(4, device=C.device), torch.tensor(6, device=C.device)))
    Cs = prune(cost_matrix(x, x, cost_type="cosine").unsqueeze(0))
    Ct = prune(cost_matrix(y, y, cost_type="cosine").unsqueeze(0))

    T, _ = GW_alg(Cs, Ct, Other_C=prune(C.unsqueeze(0)))
    print(T)