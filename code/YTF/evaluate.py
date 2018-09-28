#!/usr/bin/env python
# coding: utf-8
# author: BlancRay
import argparse

layer_num = 27
# layer_num = 64

import os
import sys
import _pickle as cPickle
import numpy as np

root_dir = os.path.dirname(os.path.abspath("__file__")) + "/../../"
sys.path.append(root_dir + "code/")
from featurer import Featurer


def cos_sim(v1, v2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA != 0.0 and normB != 0.0:
        cos = dot_product / ((normA * normB) ** 0.5)
    else:
        cos = 1.0
    sim = 0.5 + 0.5 * cos
    return sim


def norml2_sim(v1, v2):
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        normA += a ** 2
        normB += b ** 2
    normA **= 0.5
    normB **= 0.5
    diff = 0.0
    for a, b in zip(v1, v2):
        a /= normA
        b /= normB
        diff += (a - b) ** 2
    if normA != 0.0 and normB != 0.0:
        diff **= 0.5
    else:
        diff = 1.0
    sim = 1.0 - 0.5 * diff
    return sim


def get_feature(featurer, image_dir, feature_dim=512):
    final_feature = np.zeros(feature_dim)
    for name in os.listdir(image_dir):
        image_file = image_dir + name
        feature = featurer.test(image_file=image_file)
        if feature is not None:
            final_feature += feature / np.linalg.norm(feature, ord=2)
    final_feature /= np.linalg.norm(final_feature, ord=2)
    return final_feature


def test(args):
    image_dir = root_dir + "data/YTF/"

    image_dirs = set()
    pairs = list()
    for k, line in enumerate(open(image_dir + "pairs.txt")):
        item = line.strip().split()[2:]
        item[0] = image_dir + "images/" + item[0] + "/"
        item[1] = image_dir + "images/" + item[1] + "/"
        assert len(item) == 3
        pairs.append(tuple(item))
        image_dirs.add(item[0])
        image_dirs.add(item[1])

    feature_file = image_dir + "feature_%d.pkl" % layer_num
    if not os.path.exists(feature_file):
        model_dir = args.model_dir + "/"
        featurer = Featurer(deploy_prototxt=model_dir + "deploy.prototxt",
                            model_file=model_dir + "train.caffemodel",
                            mean_file=model_dir + "mean.binaryproto",
                            device_id=args.device_id,
                            ratio=args.ratio,
                            scale=args.scale,
                            resize_height=args.resize_height,
                            resize_width=args.resize_width,
                            raw_scale=args.raw_scale,
                            input_scale=args.input_scale,
                            gray=args.gray,
                            oversample=args.oversample,
                            feature_layer_names=args.feature_layer_names)
        features = dict()
        for k, image_dir in enumerate(image_dirs):
            if image_dir not in features:
                features[image_dir.replace(root_dir, "")] = get_feature(featurer, image_dir)
            print("processed:", k)
            sys.stdout.flush()
        cPickle.dump(features, open(feature_file, "wb"))
    else:
        features = cPickle.load(open(feature_file, "rb"), encoding='latin')

    sims = list()
    threds = list()
    for k, pair in enumerate(pairs):
        image_dir1, image_dir2, tag = pair[:3]
        # person1
        feature1 = features[image_dir1.replace(root_dir, "")]
        # person2
        feature2 = features[image_dir2.replace(root_dir, "")]
        # sim
        # sim = cos_sim(feature1, feature2)
        sim = norml2_sim(feature1, feature2)
        sims.append((sim, int(tag), image_dir1, image_dir2))
        threds.append(sim)

    best_accuracy = 0.0
    best_thred = 0.0
    with open(image_dir + "roc_%d.txt" % layer_num, "wb") as f:
        for thred in sorted(threds):
            tp = 0
            fn = 0
            tn = 0
            fp = 0
            for sim, tag, image_dir1, image_dir2 in sims:
                if tag == 1:
                    if sim >= thred:
                        tp += 1
                    else:
                        fn += 1
                    #    print "fp", image_dir1, image_dir2
                if tag == 0:
                    if sim < thred:
                        tn += 1
                    else:
                        fp += 1
                    #    print "fn", image_dir1, image_dir2
            tpr = 1.0 * tp / max(tp + fn, 1)
            fnr = 1.0 * fn / max(tp + fn, 1)
            tnr = 1.0 * tn / max(tn + fp, 1)
            fpr = 1.0 * fp / max(tn + fp, 1)
            accuracy = 1.0 * (tp + tn) / (tp + fp + tn + fn)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thred = thred
            f.write(b"%.6f %.6f\n" % (tpr, fpr))
            # print thred, (tp + fp + tn + fn), tpr, tnr, accuracy
        print("best:", len(pairs), best_thred, best_accuracy)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='set model dir',
                        default=root_dir + "model/ResNet-%d/" % layer_num)
    parser.add_argument('--feature_layer_names', type=list, help='set feature layer names', default=['fc5'])
    parser.add_argument('--device_id', type=int, help='set device id', default=0)
    parser.add_argument('--ratio', type=float, help='set image ratio', default=-1.0)
    parser.add_argument('--scale', type=float, help='set image scale', default=1.1)
    parser.add_argument('--resize_height', type=int, help='set image height', default=144)
    parser.add_argument('--resize_width', type=int, help='set image height', default=144)
    parser.add_argument('--raw_scale', type=float, help='set raw scale', default=255.0)
    parser.add_argument('--input_scale', type=float, help='set raw scale', default=0.0078125)
    parser.add_argument('--gray', type=bool, help='set gray', default=False)
    parser.add_argument('--oversample', type=bool, help='set oversample', default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":
    test(parse_arguments(sys.argv[1:]))
