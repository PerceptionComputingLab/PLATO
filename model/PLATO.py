# Harbin Institute of Technology Bachelor Thesis
# Author: HIT Michael_Bryant
# Mail: 1137892110@qq.com

import torch.nn as nn
import torch
import torch.distributions as td

from utils.utils import *
from models.unet import Unet


class MELT(nn.Module):
    def __init__(
        self,
        name,
        num_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.epsilon = epsilon
        conv_fn = nn.Conv2d
        # whether to use only the diagonal (independent normals)
        self.diagonal = diagonal

        self.mean_l1 = -
        self.log_cov_diag_l1 =- 
        self.cov_factor_l1 = -

        self.mean_l2 = -
        self.log_cov_diag_l2 = -
        self.cov_factor_l2 = -

    def forward(self, logits):
        batch_size = logits.shape[0]  # Get the batchsize

        # tensor size num_classesxHxW
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean1 = self.mean_l1(logits)
        cov_diag1 = self.log_cov_diag_l1(logits).exp() + self.epsilon
        mean1 = mean1.view((batch_size, -1))
        cov_diag1 = cov_diag1.view((batch_size, -1))
        cov_factor1 = self.cov_factor_l1(logits)
        cov_factor1 = cov_factor1.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor1 = cov_factor1.flatten(2, 3)
        cov_factor1 = cov_factor1.transpose(1, 2)
        cov_factor1 = cov_factor1
        cov_diag1 = cov_diag1 + self.epsilon

        mean2 = self.mean_l2(logits)
        cov_diag2 = self.log_cov_diag_l2(logits).exp() + self.epsilon
        mean2 = mean2.view((batch_size, -1))
        cov_diag2 = cov_diag2.view((batch_size, -1))
        cov_factor2 = self.cov_factor_l2(logits)
        cov_factor2 = cov_factor2.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor2 = cov_factor2.flatten(2, 3)
        cov_factor2 = cov_factor2.transpose(1, 2)
        cov_factor2 = cov_factor2
        cov_diag2 = cov_diag2 + self.epsilon

        if self.diagonal:
            base_distribution1 = td.Independent(
                td.Normal(loc=mean1, scale=torch.sqrt(cov_diag1)), 1
            )
        else:
            try:
                base_distribution1 = td.LowRankMultivariateNormal(
                    loc=mean1, cov_factor=cov_factor1, cov_diag=cov_diag1
                )
            except:
                print(
                    "Covariance became not invertible. Using independent normals for this batch!"
                )
                base_distribution1 = td.Independent(
                    td.Normal(loc=mean1, scale=torch.sqrt(cov_diag1)), 1
                )

        distribution1 = ReshapedDistribution(
            base_distribution=base_distribution1,
            new_event_shape=event_shape,
            validate_args=False,
        )

        if self.diagonal:
            base_distribution2 = td.Independent(
                td.Normal(loc=mean2, scale=torch.sqrt(cov_diag2)), 1
            )
        else:
            try:
                base_distribution2 = td.LowRankMultivariateNormal(
                    loc=mean2, cov_factor=cov_factor2, cov_diag=cov_diag2
                )
            except:
                print(
                    "Covariance became not invertible. Using independent normals for this batch!"
                )
                base_distribution2 = td.Independent(
                    td.Normal(loc=mean2, scale=torch.sqrt(cov_diag2)), 1
                )
        distribution2 = ReshapedDistribution(
            base_distribution=base_distribution2,
            new_event_shape=event_shape,
            validate_args=False,
        )

        shape = (batch_size,) + event_shape
        logit_mean1 = mean1.view(shape)
        cov_diag_view1 = cov_diag1.view(shape).detach()
        cov_factor_view1 = (
            cov_factor1.transpose(2, 1)
            .view((batch_size, self.num_classes * self.rank) + event_shape[1:])
            .detach()
        )

        logit_mean2 = mean2.view(shape)
        cov_diag_view2 = cov_diag2.view(shape).detach()
        cov_factor_view2 = (
            cov_factor2.transpose(2, 1)
            .view((batch_size, self.num_classes * self.rank) + event_shape[1:])
            .detach()
        )

        output_dict = {
            "logit_mean1": logit_mean1.detach(),
            "cov_diag1": cov_diag_view1,
            "cov_factor1": cov_factor_view1,
            "distribution1": distribution1,
            "logit_mean2": logit_mean2.detach(),
            "cov_diag2": cov_diag_view2,
            "cov_factor2": cov_factor_view2,
            "distribution2": distribution2,
        }

        return logit_mean1, logit_mean2, output_dict
