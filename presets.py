import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
#        mean=(0.485, 0.456, 0.406),
        mean=(123.68,116.78,103.94),
#        std=(0.229, 0.224, 0.225),
        std=(1.0, 1.0, 1.0),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):  
#        transforms.Resize(size=500,interpolation=interpolation)
        trans = [transforms.Resize(size=500,interpolation=interpolation),transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        
        trans.extend(
            [
                transforms.PILToTensor(),
#                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.to(torch.float)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))
        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
#        mean=(0.485, 0.456, 0.406),
        mean=(123.68,116.78,103.94),
#        std=(0.229, 0.224, 0.225),
        std=(1.0, 1.0, 1.0),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
#                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.to(torch.float)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

class ClassificationPresetCllct:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
#        mean=(0.485, 0.456, 0.406),
        mean=(123.68,116.78,103.94),
#        std=(0.229, 0.224, 0.225),
        std=(1.0, 1.0, 1.0),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
#                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.to(torch.float)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
