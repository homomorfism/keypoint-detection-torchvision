def concatenate_lists(*args):
    concatenated: list[dict] = [{} for _ in range(len(args[0]))]

    for data in args:
        for ii in range(len(data)):
            concatenated[ii] |= data[ii]

    return concatenated


def make_grid_with_keypoints(images, true_keypoints, pred_keypoints):
    # https://discuss.pytorch.org/t/add-label-captions-to-make-grid/42863/4
    pass
