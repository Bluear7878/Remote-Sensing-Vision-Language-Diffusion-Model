import numpy as np


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f'Only support 4D, 3D, and 2D tensor. But got: {n_dim}D')
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)
