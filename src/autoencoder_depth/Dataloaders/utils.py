import torchvision.transforms as transforms
import torch
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


def default_transform(scale_size=None):
    normalize = __imagenet_stats

    rgb_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size is not None:
        rgb_list.append(transforms.Resize(scale_size))

    return transforms.Compose(rgb_list)


def inception_color_preproccess(scale_size=None, normalize=None):
    if normalize is None:
        normalize = __imagenet_stats

    tf = [transforms.ToTensor(),
          ColorJitter(
              brightness=0.4,
              contrast=0.4,
              saturation=0.4,
          ),
          Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
          transforms.Normalize(**normalize), ]

    if scale_size is not None:
        tf.append(transforms.Resize(scale_size))
    return transforms.Compose(tf)


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        super(ColorJitter, self).__init__()
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs = 0.299 * img[0] + 0.587 * gs[1] + 0.114 * gs[2]
        gs.unsqueeze(0).repeat(3, 1, 1)
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)
