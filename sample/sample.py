import sys
import os
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from EndPts import EndPts
import cv2

if __name__ == '__main__':
    # Set seed
    np.random.seed(1020)
    # Set parameter
    n1 = 400
    n2 = 10
    theta = np.linspace(0, 2 * np.pi, n2)
    t = np.linspace(0, 1, n2)
    t_c = np.copy(t)
    theta, t = np.meshgrid(theta, t)
    theta, t = theta.flatten(), t.flatten()
    radius = 2.5

    # Set control points
    ctrl1 = np.array([[-10, 10, 100], [0, 10, 120], [70, 70, 70], [100, 100, 0]])
    diff1 = np.vstack([np.zeros(shape=(1, 3)), np.diff(ctrl1, axis=0)])
    t1 = np.linalg.norm(diff1, axis=1)
    t1 /= t1[-1]

    # fit
    curve1 = np.polyfit(t1, ctrl1, deg=3)

    # generate
    x = np.polyval(curve1[:, 0], t) + radius * np.cos(theta)
    y = np.polyval(curve1[:, 1], t) + radius * np.sin(theta)
    z = np.polyval(curve1[:, 2], t)
    cont = np.vstack([x, y, z])
    tri = mtri.Triangulation(theta, t)

    # copy contour with random error
    n_cont = 5
    traj = np.zeros(shape=(n_cont, cont.shape[0], cont.shape[1]), dtype=np.float)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_cont):
        th = np.random.uniform(0, np.pi / 4)
        c = np.cos(th)
        s = np.sin(th)
        scale = np.diag(np.random.uniform(0.8, 1.2, 3))
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        shift = np.array([400 + np.random.uniform(-50, 50), 400 + np.random.uniform(-50, 50), np.random.uniform(-10, 100)])
        cont_ = scale @ rot @ cont + shift.reshape(3, 1)
        traj[i] = cont_
        ax.plot_trisurf(cont_[0], cont_[1], cont_[2], triangles=tri.triangles, cmap=plt.cm.Spectral)
    plt.savefig("ground-truth.png")

    # create folder
    if not os.path.exists("view_00"):
        os.mkdir("view_00")
        os.mkdir(os.path.join("view_00", "wire"))
    if not os.path.exists("view_01"):
        os.mkdir("view_01")
        os.mkdir(os.path.join("view_01", "wire"))

    # generate segmentation
    camera1 = np.array([[0.7, 0, 0], [0, 0.7, 0]])
    camera2 = 2.4 * np.array([[0.249766, -0.00197158, 0.064797], [0.00170334, 0.25022, 0.0645517]])
    img1 = np.zeros(shape=(512, 512), dtype=np.int)
    img2 = np.zeros(shape=(512, 512), dtype=np.int)
    for i, j, k in tri.triangles:
        for c in range(n_cont):
            pts = np.array([traj[c, :, i], traj[c, :, j], traj[c, :, k]], dtype=np.float)
            # camera 1
            pts_ = (camera1 @ pts.transpose()).transpose()
            pts_ = np.array(pts_, dtype=np.int).reshape(-1, 1, 2)
            cv2.fillConvexPoly(img1, pts_, color=255)
            # camera 2
            pts_ = (camera2 @ pts.transpose()).transpose()
            pts_ = np.array(pts_, dtype=np.int).reshape(-1, 1, 2)
            cv2.fillConvexPoly(img2, pts_, color=255)
    cv2.imwrite(os.path.join("view_00", "energy.png"), img1)
    cv2.imwrite(os.path.join("view_01", "energy.png"), img2)

    # Affine
    affine = np.zeros(shape=(3, 4), dtype=np.float)
    affine[2, 3] = 1.0
    affine[:2, :3] = camera1
    np.save(os.path.join("view_00", "projection.npy"), affine)
    affine[2, 3] = 1.0
    affine[:2, :3] = camera2
    np.save(os.path.join("view_01", "projection.npy"), affine)

    # Bond
    for i in range(n_cont):
        endpts = EndPts()
        traj_ = camera1 @ traj[i]
        endpts.wire_id = 0
        endpts.first_bond = traj_[:, 0].tolist()
        endpts.first_radius = 1
        endpts.second_bond[0] = traj_[:, -1].tolist()
        endpts.second_radius = 1
        endpts.wedge_neck = endpts.second_bond
        endpts.wire_width = 14
        endpts.dump_json(os.path.join("view_00", "wire", "wire_" + str(i).zfill(2) + ".json"))

    for i in range(n_cont):
        endpts = EndPts()
        traj_ = camera2 @ traj[i]
        endpts.wire_id = 0
        endpts.first_bond = traj_[:, 0].tolist()
        endpts.first_radius = 1
        endpts.second_bond[0] = traj_[:, -1].tolist()
        endpts.second_radius = 1
        endpts.wedge_neck = endpts.second_bond
        endpts.wire_width = 14
        endpts.dump_json(os.path.join("view_01", "wire", "wire_" + str(i).zfill(2) + ".json"))






