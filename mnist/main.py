import torch.optim as optim
import torch.nn as nn
from model import Model
from torch.utils.tensorboard import SummaryWriter
from mnist.config import Config
from mnist.learningProcess import LearningProcess
from analizer.gradientLogger import GradientLogger


if __name__ == "__main__":
    model = Model()
    config = Config()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(config.pathToLogs)
    logger = GradientLogger(model, writer)
    learner = LearningProcess(optimizer, criterion, logger, writer)
    learner.train(model)
    learner.validate(model)
    learner.test(model)