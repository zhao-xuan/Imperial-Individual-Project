_base_ = "seresnet/seresnet101_8xb32_in1k.py"

dataset_type = 'CustomDataset'
classes = ['non_nodule', 'nodule']  # The category names of your dataset

model = dict(head=dict(num_classes=2, topk=(1,)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_prefix='../local-global/data/luna16'

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix + "/train_gen_ann.txt", # or /train_ann.txt
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix + "/test_ann.txt",
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix + "/test_ann.txt",
        classes=classes
    )
)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=-1)
)

evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=(1,)))
checkpoint_config = dict(interval=-1)
