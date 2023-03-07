import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from preprocess import mean, std
from settings import img_size

normalize = transforms.Normalize(mean=mean, std=std)


def run_model_on_batch(
        model: torch.nn.Module,
        batch: torch.Tensor,
):
    _, patch_distances = model.push_forward(batch)

    # get model prediction
    min_distances = -nn.functional.max_pool2d(-patch_distances,
                                              kernel_size=(patch_distances.size()[2],
                                                           patch_distances.size()[3]))
    min_distances = min_distances.view(-1, model.num_prototypes)
    prototype_activations = model.distance_2_similarity(min_distances)
    predicted_cls = torch.argmax(torch.softmax(model.last_layer(prototype_activations), dim=-1), dim=-1)
    patch_activations = model.distance_2_similarity(patch_distances).cpu().detach().numpy()

    return predicted_cls.cpu().detach().numpy(), patch_activations


def run_model_on_dataset(
        model: nn.Module,
        dataset: Dataset,
        num_workers: int,
        batch_size: int
):
    """
    Runs the model on all images in the given directory and saves the results.
    :param model: the model to run
    :param dataset: pytorch dataset
    :param num_workers: number of parallel workers for the DataLoader
    :param batch_size: batch size for the DataLoader
    :return a generator of model outputs for each of the images, together with batch data
    """
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    current_idx = 0

    for img_tensor, target in test_loader:
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        batch_samples = dataset.samples[current_idx:current_idx + batch_size]
        batch_filenames = [os.path.basename(s[0]) for s in batch_samples]
        with torch.no_grad():
            predicted_cls, patch_activations = run_model_on_batch(
                model=model, batch=img_tensor
            )
            current_idx += img_tensor.shape[0]

        img_numpy = img_tensor.clone().cpu().detach().numpy()
        for d in range(3):
            img_numpy[:, d] = (img_numpy[:, d] * std[d] + mean[d])

        yield {
            'filenames': batch_filenames,
            'target': target.cpu().detach().numpy(),
            'img_tensor': img_tensor,
            'img_original_numpy': img_numpy,
            'patch_activations': patch_activations,
            'predicted_cls': predicted_cls,
        }
