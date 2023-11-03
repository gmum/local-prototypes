from typing import Optional

import numpy as np
import torch
from torch import nn


class PPNetAdversarialWrapper(nn.Module):
    """
    Wrapper over the PPNet model that allows for adversarially attack activations of selected prototypes,
    over a selected image, and with a selected mask.
    The attack aims to minimize the activation of the selected prototypes, while modifying only the masked pixels.
    """

    def __init__(
            self,
            model: nn.Module,
            proto_nums: Optional[np.ndarray] = None,
            img: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            focal_sim: bool = False
    ):
        """
        :param model: PPNet model
        :param img: an image to attack
        :param proto_nums: vector of prototype numbers to attack
        :param mask: binary mask, 1 for pixels that can be modified, 0 for pixels that cannot be modified
        """
        super(PPNetAdversarialWrapper, self).__init__()
        self.model = model
        self.proto_nums = proto_nums
        self.mask = mask
        self.focal_sim = focal_sim

        assert (img is None and mask is None) or (img is not None and mask is not None)

        if img is not None and mask is not None:
            # ensure that we do not propagate gradients through the image and the mask
            self.img = img.clone()
            self.img.requires_grad = False
            # self.mask = torch.tensor(mask, device=self.img.device)
            self.mask.requires_grad = False
        else:
            self.img = None
            self.mask = None

        self.initial_activation, self.final_activation = None, None

    def forward(self, x):
        if self.img is None:
            x2 = x
        else:
            # 'x' can be modified by cleverhans
            # 'x2' is the actual output image.
            # We use masking to ensure that cleverhans can affect only the masked pixels.
            x2 = x * self.mask + self.img * (1 - self.mask)

        conv_output, distances = self.model.push_forward(x2)
        if self.proto_nums is not None:
            distances = distances[:, self.proto_nums]

        activations = self.model.distance_2_similarity(distances).flatten(start_dim=2)
        max_activations, _ = torch.max(activations, dim=-1)
        final_activations = max_activations[0].clone().cpu().detach().numpy()
        self.final_activation = final_activations
        if self.initial_activation is None:
            self.initial_activation = final_activations

        if self.focal_sim:
            distances = distances.flatten(start_dim=2)
            mean_dist = torch.mean(distances, dim=-1).unsqueeze(-1)
            min_dist, _ = torch.min(distances, dim=-1)

            sim_diff = self.model.distance_2_similarity(min_dist) - self.model.distance_2_similarity(mean_dist)
            return torch.mean(sim_diff).unsqueeze(0).unsqueeze(0)
        else:
            return torch.mean(max_activations).unsqueeze(0).unsqueeze(0)
