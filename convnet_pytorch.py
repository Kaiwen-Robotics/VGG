"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem


    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    """
    self.conv1 = nn.Sequential(nn.Conv2d( 3, n_channels, 3, stride=1,padding=1),
                               nn.BatchNorm2d(n_channels),
                               nn.ReLU(True))
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Sequential(nn.Conv2d(n_channels, n_channels*2, 3,stride=1, padding=1),
                               nn.BatchNorm2d(n_channels*2),
                               nn.ReLU(True))
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv3_a = nn.Sequential(nn.Conv2d(n_channels*2, n_channels * 4, 3, stride=1,padding=1),
                               nn.BatchNorm2d(n_channels * 4),
                               nn.ReLU(True))
    self.conv3_b = nn.Sequential(nn.Conv2d(n_channels * 4, n_channels * 4, 3, stride=1,padding=1),
                                 nn.BatchNorm2d(n_channels * 4),
                                 nn.ReLU(True))
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv4_a = nn.Sequential(nn.Conv2d(n_channels * 4, n_channels * 8, 3, stride=1,padding=1),
                                 nn.BatchNorm2d(n_channels * 8),
                                 nn.ReLU(True))
    self.conv4_b = nn.Sequential(nn.Conv2d(n_channels * 8, n_channels * 8, 3, stride=1,padding=1),
                                 nn.BatchNorm2d(n_channels * 8),
                                 nn.ReLU(True))
    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

    self.conv5_a = nn.Sequential(nn.Conv2d(n_channels * 8, n_channels * 8, 3,stride=1, padding=1),
                                 nn.BatchNorm2d(n_channels * 8),
                                 nn.ReLU(True))
    self.conv5_b = nn.Sequential(nn.Conv2d(n_channels * 8, n_channels * 8, 3, stride=1,padding=1),
                                 nn.BatchNorm2d(n_channels * 8),
                                 nn.ReLU(True))
    self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.linear = nn.Linear(n_channels * 8, n_classes)
    """
    self.conv1 = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(True))
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(True))
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv3_a = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(True))
    self.conv3_b = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv4_a = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True))
    self.conv4_b = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True))
    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv5_a = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True))
    self.conv5_b = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(True))
    self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.linear = nn.Linear(in_features=512, out_features=n_classes)

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = x.size(0)
    out = self.conv1(x)
    out = self.maxpool1(out)
    out = self.conv2(out)
    out = self.maxpool2(out)
    out = self.conv3_a(out)
    out = self.conv3_b(out)
    out = self.maxpool3(out)
    out = self.conv4_a(out)
    out = self.conv4_b(out)
    out = self.maxpool4(out)
    out = self.conv5_a(out)
    out = self.conv5_b(out)
    out = self.maxpool5(out).view(batch_size,-1)
    out = self.linear(out)

    #raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
