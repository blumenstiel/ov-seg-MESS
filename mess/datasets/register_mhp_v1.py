import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

CLASSES = ('others',
           'hat',
           'hair',
           'sunglasses',
           'upper clothes',
           'skirt',
           'pants',
           'dress',
           'belt',
           'left shoe',
           'right shoe',
           'face',
           'left leg',
           'right leg',
           'left arm',
           'right arm',
           'bag',
           'scarf',
           'torso skin'
           )


def register_dataset(root):
    ds_name = 'mhp_v1'
    root = os.path.join(root, 'LV-MHP-v1')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('train', 'images_detectron2/train', 'annotations_detectron2/train', CLASSES),
        ('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        ('test', 'images_detectron2/test', 'annotations_detectron2/test', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpg'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
