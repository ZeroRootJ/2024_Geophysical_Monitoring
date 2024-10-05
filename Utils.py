# Class for index conversion between 1D and 2D indices
class IdxConv:
    def __init__(self, nx):
        self.nx = nx

    # Convert (x, z) to 1D index
    def flatten_index(self, x, z):
        return z * self.nx + x

    # Convert 1D index back to (x, z)
    def unflatten_index(self, index):
        x = index % self.nx
        z = index // self.nx
        return x, z