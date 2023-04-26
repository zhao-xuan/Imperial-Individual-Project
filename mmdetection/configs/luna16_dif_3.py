# The new config inherits a base config to highlight the necessary modification
# _base_ = 'mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py'
# _base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
_base_ = 'faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
# _base_ = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        # mask_head=dict(num_classes=1),
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

img_prefix = '../data/luna16_sliced'

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nodule',)
data = dict(
    train=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file=img_prefix + '/train_kfold_diffusion_3.json',
        pipeline=train_pipeline
    ),
    val=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file=img_prefix + '/test_kfold_diffusion_3.json',
        pipeline=test_pipeline
    ),
    test=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file=img_prefix + '/test_kfold_diffusion_3.json',
        pipeline=test_pipeline
    )
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(interval=1, metric='bbox')

# load_from = "work_dirs/old_config/latest.pth"
