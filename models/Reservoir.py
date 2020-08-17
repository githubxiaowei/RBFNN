
from networkx import erdos_renyi_graph, adjacency_matrix
from utils import *


class ReservoirEncoder:
    def __init__(self, reservoirConf):
        self.nz = reservoirConf.nz
        self.nu = reservoirConf.nu
        self.alpha = reservoirConf.alpha
        self.target_rho = reservoirConf.target_rho
        input_scale = reservoirConf.input_scale
        self.state = None
        self.activation = reservoirConf.activation

        # sparse recurrent weights init
        if reservoirConf.connectivity < 1:
            g = erdos_renyi_graph(
                reservoirConf.nz,
                reservoirConf.connectivity,
                seed=42,
                directed=True
            )
            self.A = np.array(adjacency_matrix(g).todense()).astype(np.float)

        # full-connected recurrent weights init
        else:
            self.A = np.random.uniform(-1, +1, size=(self.nz, self.nz))
            self.A = (self.A + self.A.T)/2

        rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= self.target_rho / rho
        self.B = np.random.uniform(-input_scale, input_scale, size=(self.nz, self.nu))

    def state_transition(self, z, u):
        z = (1 - self.alpha) * z + self.alpha * self.activation(self.A @ z + self.B @ u)
        return z

    def transform(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nx // self.nu):
            u = x[i * self.nu:(i + 1) * self.nu]
            z = self.state_transition(z, u)
        return z

    def echostate(self, x):
        nx, nt = x.shape
        z = np.zeros((self.nz, nt))
        for i in range(nt):
            u = x[-self.nu:, i]
            z[:, i] = self.state_transition(z[:, i-1], u)
        return z
