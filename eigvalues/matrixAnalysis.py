import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config

class RandomNormalMatrix:
    def __init__(self, n: int, m: int, mean: float = 0.0, std: float = 1.0):
        self.n = n
        self.m = m
        self.mean = mean
        self.std = std
        self.matrix = self._generateMatrix(n, m, mean, std)
        self.eigenvalues = self._eig(self.matrix)

    def _generateMatrix(self, n: int, m: int, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        generator = torch.normal(mean=mean, std=std, size=(n, m))
        return generator
    
    def _eig(self, tensor: torch.tensor):
        eigenvalues, eigenvectors = torch.linalg.eig(tensor)
        return eigenvalues.real


def getTheoryBoundary(n: int, m: int, mean: float = 0.0, std: float = 1.0) -> tuple:
    left = (n * mean) - (2 * std * (n ** 0.5))
    right = (n * mean) + (2 * std * (n ** 0.5))
    return left, right


def test(config: Config = Config()) -> None:
    matrices = []
    for _ in range(config.n_samples):
        matrices.append(RandomNormalMatrix(config.n, config.m, config.mean, config.std))
    container = []
    for matrix in matrices:
        container.append(matrix.eigenvalues)
    print("total number of checked matrices: " + str(len(container)))
    container = torch.cat(container, dim=0)
    print("left boundary:" + str(getTheoryBoundary(config.n, config.m, config.mean, config.std)[0]))
    print("right boundary:" + str(getTheoryBoundary(config.n, config.m, config.mean, config.std)[1]))
    print("min eigvalue: " + str(torch.min(container)))
    print("max eigvalue: " + str(torch.max(container)))
    # if config.plot:
    writer = SummaryWriter(config.pathToLogs)
    writer.add_histogram('eigvalues', container, 0)
    writer.close()

if __name__ == "__main__":
    test()