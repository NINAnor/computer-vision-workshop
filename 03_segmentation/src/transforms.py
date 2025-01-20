import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms

albumentations_transform = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

# Torchvision resize transformation with InterpolationMode.BILINEAR
resize_transform = transforms.Compose(
    [
        transforms.Resize(
            (512, 512),
            interpolation=Image.HAMMING,  # NEAREST / BILINEAR / BICUBIC / BOX / HAMMING / LANCZOS
        ),
    ]
)