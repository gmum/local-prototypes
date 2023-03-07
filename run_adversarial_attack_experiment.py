"""
An experiment that attacks activation of prototypes adversarially.
Its goal is to determine whether the prototypes are local.
If they are local, the adversarial attack should not be successful.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from adversarial_attacks.adversarial_attack import attack_images_target_class_prototypes
from adversarial_attacks.run_on_dataset import run_model_on_image_folder, run_model_on_batch
from settings import results_dir, test_dir, img_size


def get_heatmap(patch_activation: np.ndarray) -> np.ndarray:
    upsampled_activation = cv2.resize(patch_activation,
                                      dsize=(img_size, img_size),
                                      interpolation=cv2.INTER_CUBIC)
    rescaled_act = upsampled_activation - np.amin(upsampled_activation)
    rescaled_act = rescaled_act / np.amax(rescaled_act)

    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap[..., ::-1]


def get_mean_heatmap(patch_activations: np.ndarray, proto_nums: np.ndarray) -> np.ndarray:
    mean_activation = np.mean(patch_activations[proto_nums], axis=0)
    return get_heatmap(mean_activation)


def run_adversarial_attack_on_prototypes(args):
    for ch_path, model_key in tqdm(zip(args.model_checkpoints, args.model_keys),
                                   desc=f'running experiment on {len(args.model_checkpoints)} models'):
        print(f'Loading model {model_key} from {ch_path}...')
        if torch.cuda.is_available():
            model = torch.load(ch_path).cuda()
        else:
            model = torch.load(ch_path, map_location=torch.device('cpu'))

        output_dir = os.path.join(results_dir, 'adversarial_attack', model_key)
        output_adv_img_dir = os.path.join(output_dir, 'adversarial_images')
        os.makedirs(output_adv_img_dir, exist_ok=True)

        pbar = tqdm()

        n_samples, n_correct_before, n_correct_after = 0, 0, 0
        for batch_result in run_model_on_image_folder(
                model=model,
                directory=test_dir,
                num_workers=args.n_jobs,
                batch_size=args.batch_size
        ):
            adversarial_result = attack_images_target_class_prototypes(
                model=model,
                img=batch_result['img_tensor'],
                activations=batch_result['patch_activations'],
                cls=batch_result['target'],
                epsilon=args.epsilon,
                epsilon_iter=args.epsilon_iter,
                nb_iter=args.nb_iter,
            )

            n_samples += len(batch_result['filenames'])
            n_correct_before += np.sum(batch_result['predicted_cls'] == batch_result['target'])

            with torch.no_grad():
                predicted_cls_adv, patch_activations_adv = run_model_on_batch(
                    model=model, batch=adversarial_result['img_modified_tensor']
                )

            n_correct_after += np.sum(predicted_cls_adv == batch_result['target'])

            figsize = 10, 10
            extension = 'jpg'
            alpha = 0.7
            for i in range(len(batch_result['filenames'])):
                filename = batch_result['filenames'][i]
                img_original = batch_result['img_original_numpy'][i]
                img_modified = adversarial_result['img_modified_numpy'][i]
                sample_mask = adversarial_result['mask'][i]
                proto_nums = adversarial_result['proto_nums'][i]

                img_original = img_original.transpose(1, 2, 0)
                img_modified = img_modified.transpose(1, 2, 0)
                sample_mask = sample_mask.transpose(1, 2, 0)
                with_mask = img_modified * sample_mask + \
                            (1 - sample_mask) * (alpha * sample_mask + (1 - alpha) * img_modified)

                heatmap = get_mean_heatmap(batch_result['patch_activations'][i], proto_nums)
                overlayed_img_original = 0.5 * img_original + 0.3 * heatmap

                heatmap_modified = get_mean_heatmap(patch_activations_adv[i], proto_nums)
                overlayed_img_modified = 0.5 * img_modified + 0.3 * heatmap_modified

                for im, desc in zip(
                        [img_modified, img_original, with_mask, overlayed_img_original, overlayed_img_modified],
                        ['modified', 'original', 'modified_with_mask', 'original_heatmap', 'modified_heatmap']):
                    plt.figure(figsize=figsize)
                    plt.imshow(im, vmin=0, vmax=1)
                    plt.axis('off')
                    plt.savefig(os.path.join(output_adv_img_dir,
                                             filename.replace(f'.{extension}', f'_{desc}.{extension}')),
                                bbox_inches='tight', pad_inches=0)
                    plt.close()
            acc1 = n_correct_before / n_samples * 100
            acc2 = n_correct_after / n_samples * 100
            pbar.set_description('Running model + attack. Accuracy before: {:.2f}%, after: {:.2f}%)'.format(acc1, acc2))
            pbar.update()

        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarially attack prototypes')
    parser.add_argument('model_checkpoints', nargs='+', type=str,
                        help='Paths to the checkpoints (.pth files) of the evaluated models')
    parser.add_argument('model_keys', nargs='+', type=str, help='Names for the models to display in plot titles')

    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs (for DataLoader)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for using the model')

    # parameters for the adversarial attack
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Maximum perturbation of the adversarial attack')
    parser.add_argument('--epsilon_iter', type=float, default=0.01,
                        help='Maximum perturbation of the adversarial attack within one iteration')
    parser.add_argument('--nb_iter', type=iter, default=40,
                        help='Number of iterations of the adversarial attack')

    run_adversarial_attack_on_prototypes(parser.parse_args())
