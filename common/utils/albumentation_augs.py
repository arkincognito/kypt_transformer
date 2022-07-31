import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
from matplotlib import pyplot as plt


p1 = 0.15
p2 = 0.35

augmentation_transform = A.Compose(
    [
        A.MotionBlur(always_apply=False, p=p2, blur_limit=(3, 25)),
        A.OneOf(
            [
                A.GaussianBlur(p=0.35),
                A.GlassBlur(p=0.15),
                A.Blur(p=0.35),
                A.MedianBlur(p=0.15),
            ]
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    p=0.5,
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    brightness_by_max=True,
                ),
                A.RandomGamma(p=0.5, gamma_limit=(30, 140), eps=1e-07),
                A.CLAHE(p=0.5, clip_limit=(1, 4), tile_grid_size=(14, 14)),
            ],
            p=p1,
        ),
        A.RGBShift(always_apply=False, p=p2, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
        A.OneOf(
            [
                A.ISONoise(p=0.5, intensity=(0.1, 0.3), color_shift=(0.02, 0.15)),
                A.GaussNoise(p=0.5, var_limit=(10.0, 500.0)),
            ],
            p=p1,
        ),
        A.OneOf(
            [
                A.ImageCompression(always_apply=False, p=0.5, quality_lower=15, quality_upper=70, compression_type=0),
                A.Downscale(0.25, 0.9, p=0.5),
            ],
            p=p2,
        ),
        A.CoarseDropout(
            always_apply=False,
            p=p1,
            max_holes=16,
            max_height=8,
            max_width=8,
            min_holes=2,
            min_height=2,
            min_width=2,
        ),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
all_transforms = A.Compose(
    [
        A.Equalize(p=0.15),
        A.GaussNoise(p=0.15),
        A.GaussianBlur(p=0.35),
        A.GlassBlur(p=0.15),
        A.Blur(p=0.35),
        A.MedianBlur(p=0.15),
        A.MotionBlur(p=0.35),
        A.HueSaturationValue(p=0.15),
        A.ISONoise(p=0.35),
        A.ImageCompression(40, 90, p=0.15),
        A.InvertImg(p=0.35),
        A.MultiplicativeNoise((0.2, 2.0), p=0.15),
        A.RGBShift(p=0.35),
        A.RandomBrightnessContrast(p=0.35),
        A.RandomGamma(p=0.15),
        A.RandomRain(p=0.15),
        A.RandomShadow(p=0.15),
        A.RandomSnow(p=0.35),
        A.RandomSunFlare(src_radius=150, p=0.15),
        A.Solarize(p=0.35),
        A.ToGray(p=0.15),
        A.ToSepia(p=0.15),
        A.Downscale(p=0.35),
        A.ColorJitter(p=0.35),
        A.ChannelShuffle(p=0.35),
        A.ChannelDropout(p=0.15),
        A.CLAHE(clip_limit=10.0, p=0.15),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
