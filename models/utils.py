import cv2
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    _images = [torch.randn(1, 256, 256) for _ in range(3)]
    _labels = [torch.randint(0, 256, size=(1, 4, 2)) for _ in range(3)]
    _pred = [torch.randint(0, 256, size=(1, 3, 2)) for _ in range(3)]

    grid = make_grid_with_keypoints(_images, _labels, _pred)
    grid = torch.permute(grid, dims=(1, 2, 0)).numpy()

    plt.imshow(grid)
    plt.show()
