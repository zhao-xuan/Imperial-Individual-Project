
# The new config inherits a base config to highlight the necessary modification
_base_ = 'faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
    )
)

img_prefix = '../luna-16-seg-diff-data/combined'

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nodule',)
data = dict(
    train=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file='./annotation_coco_train.json'),
    val=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file='./annotation_coco_test.json'),
    test=dict(
        img_prefix=img_prefix,
        classes=classes,
        ann_file='./annotation_coco_test.json'))


runner = dict(type='EpochBasedRunner', max_epochs=48)
