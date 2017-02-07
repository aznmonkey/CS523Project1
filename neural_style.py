# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os
import skimage.io
import numpy as np
import scipy.misc
import tensorflow as tf
from stylize import stylize

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(options.content)
 ##   print(size(content_image))
    style_images = [imread(style) for style in options.styles]

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    for iteration, image in stylize(
        network=options.network,
        initial=initial,
        content=content_image,
        styles=style_images,
        iterations=options.iterations,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        output_file = None
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output

        #print(output_file)
        if output_file:
            imsave(output_file, image)
            combine_image(options.content, output_file)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def concat_images(imga, imgb):
        """
        Combines two color image ndarrays side-by-side.
        """
        ha, wa = imga.shape[:2]
        hb, wb = imgb.shape[:2]
        print(imga.shape)
        print(imgb.shape)
        print(ha)
        print(wa)
        print(hb)
        print(wa)
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
        return new_img

def rgb2yuv(rgb):
        """
        Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
        """
        rgb2yuv_filter = tf.constant(
            [[[[0.299, -0.169, 0.499],
               [0.587, -0.331, -0.418],
               [0.114, 0.499, -0.0813]]]])
        rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
        print(rgb)
     #   print(rgb2yuv_filter)
        temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
        temp = tf.nn.bias_add(temp, rgb2yuv_bias)

        return temp

def yuv2rgb(yuv):
        """
        Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
        """
        yuv = tf.mul(yuv, 255)
        yuv2rgb_filter = tf.constant(
            [[[[1., 1., 1.],
               [0., -0.34413999, 1.77199996],
               [1.40199995, -0.71414, 0.]]]])
        yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
        temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
        temp = tf.nn.bias_add(temp, yuv2rgb_bias)
        temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
        temp = tf.minimum(temp, tf.mul(
            tf.ones(temp.get_shape(), dtype=tf.float32), 255))
        temp = tf.div(temp, 255)
        return temp

def combine_image(original, styled):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('original', original, 'Original Image')
    flags.DEFINE_string('styled', styled, 'Styled Image')

    original_image = skimage.io.imread(FLAGS.original) / 255.0
    original_image = original_image.reshape((1,)+original_image.shape)
    styled_image = skimage.io.imread(FLAGS.styled) / 255.0

 ##  print(styled_image.shape)
    styled_image = styled_image.reshape((1,)+styled_image.shape)

    original = tf.placeholder("float", original_image.shape)
    styled = tf.placeholder("float", styled_image.shape)

    #print(styled)
    styled_grayscale = tf.image.rgb_to_grayscale(styled)
    #print(styled_grayscale)
    styled_grayscale_rgb = tf.image.grayscale_to_rgb(styled_grayscale)
    #print(styled_grayscale_rgb)
    styled_grayscale_yuv = rgb2yuv(styled_grayscale_rgb)

    print(original)
    original_yuv = rgb2yuv(original)

    combined_yuv = tf.concat(3, [tf.split(3, 3, styled_grayscale_yuv)[0], tf.split(3, 3, original_yuv)[1],
                                 tf.split(3, 3, original_yuv)[2]])
    combined_rbg = yuv2rgb(combined_yuv)

    init = tf.initialize_all_variables()



    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())


        combined_rbg_ = sess.run(combined_rbg, feed_dict={original: original_image, styled: styled_image})

        summary_image = concat_images(original_image.reshape(original_image.shape[1:]), styled_image.reshape(styled_image.shape[1:]))
        summary_image = concat_images(summary_image, combined_rbg_[0])
        scipy.misc.imsave("results.jpg", summary_image)

if __name__ == '__main__':
    main()

