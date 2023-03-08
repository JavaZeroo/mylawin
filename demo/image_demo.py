# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

CLASSES = ( "WATER",
            "ASPHALT",
            "GRASS",
            "HUMAN",
            "ANIMAL",
            "HIGH_VEGETATION",
            "GROUND_VEHICLE",
            "FACADE",
            "WIRE",
            "GARDEN_FURNITURE",
            "CONCRETE",
            "ROOF",
            "GRAVEL",
            "SOIL",
            "PRIMEAIR_PATTERN",
            "SNOW")
            
PALETTE = \
    ([ 148, 218, 255 ],  # light blue
    [  85,  85,  85 ],  # almost black
    [ 200, 219, 190 ],  # light green
    [ 166, 133, 226 ],  # purple    
    [ 255, 171, 225 ],  # pink
    [  40, 150, 114 ],  # green
    [ 234, 144, 133 ],  # orange
    [  89,  82,  96 ],  # dark gray
    [ 255, 255,   0 ],  # yellow
    [ 110,  87, 121 ],  # dark purple
    [ 205, 201, 195 ],  # light gray
    [ 212,  80, 121 ],  # medium red
    [ 159, 135, 114 ],  # light brown
    [ 102,  90,  72 ],  # dark brown
    [ 255, 255, 102 ],  # bright yellow
    [ 251, 247, 240 ])  # almost white



def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        PALETTE,
        opacity=args.opacity,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
