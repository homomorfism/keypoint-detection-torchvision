import cv2
import numpy as np
import torch
import torchvision.utils
from skimage.color import gray2rgb


def concatenate_lists(*args):
    concatenated: list[list] = [[] for _ in range(len(args[0]))]

    for data in args:
        for ii in range(len(data)):
            concatenated[ii].append(data[ii])

    return concatenated


def draw_keypoints(image: np.array, points: np.array, color='c'):
    assert color == 'c' or color == 'g'
    points = points.reshape(-1, 2)

    if points.shape[0] == 0:
        # No keypoints are in points
        return image

    color = (0, 255 / 256, 0) if color == 'g' else (255 / 256, 0, 0)

    for x, y in points:
        image = cv2.circle(img=image, radius=5, color=color, thickness=-1, center=(int(x), int(y)))

    return image


def make_grid_with_keypoints(images: list,
                             true_keypoints: list,
                             pred_keypoints: list):
    """
    Green is labels, red is prediction
    :param images:
    :param true_keypoints:
    :param pred_keypoints:
    :return: torch.tensor(3, _, _)
    """

    preprocessed_images = []
    for image, true, pred in zip(images, true_keypoints, pred_keypoints):
        image = image.numpy().squeeze()

        image = gray2rgb(image)
        true = true.numpy()
        pred = pred.numpy()
        image = draw_keypoints(image, true[:, :, :2], color='g')
        image = draw_keypoints(image, pred[:, :, :2], color='c')

        image = np.transpose(image, axes=(2, 0, 1))
        image = torch.from_numpy(image)

        preprocessed_images.append(image)

    return torchvision.utils.make_grid(preprocessed_images)


