import os
import random
import numpy as np
import tensorflow as tf


class data_loader:
    def __init__(self, path_natural_train, path_expression_train, path_natural_test, path_expression_test):
        #train img path
        self.path_natural_train, self.path_expression_train = self.data_dir(path_natural_train), self.data_dir(path_expression_train)
        #train label
        self.label_natural_train = self.build_ck_label(self.path_natural_train, label_type='natural')
        self.label_expression_train = self.build_ck_label(self.path_expression_train, label_type='expression')

        #test img path
        self.path_natural_test, self.path_expression_test = self.data_dir(path_natural_test), self.data_dir(path_expression_test)
        #test label
        self.label_natural_test = self.build_ck_label(self.path_natural_test, label_type='natural')
        self.label_expression_test = self.build_ck_label(self.path_expression_test, label_type='expression')

    def build_dataset(self):
        AUTO = tf.data.AUTOTUNE
        #build datalist
        nat_img, nat_label, exp_img, exp_label = self.build_pair_data()
        nat_img1, nat_label1, exp_img1, exp_label1 = self.build_oneimg_pair_data()
        #build tensorflow dataset
        #CK dataset
        nat2exp_dataset = tf.data.Dataset.from_tensor_slices((nat_img, nat_label, exp_label, exp_img)).shuffle(128 * 1000).prefetch(AUTO).batch(32)
        exp2nat_dataset = tf.data.Dataset.from_tensor_slices((exp_img, exp_label, nat_label, nat_img)).shuffle(128 * 1000).prefetch(AUTO).batch(32)
        total_dataset = tf.data.Dataset.from_tensor_slices((exp_img+nat_img, exp_label+nat_label, nat_label+exp_label, nat_img+exp_img)).shuffle(128 * 1000).prefetch(AUTO).batch(32)

        test_dataset = tf.data.Dataset.from_tensor_slices((nat_img1, nat_label1, exp_img1, exp_label1)).prefetch(AUTO).batch(32)

        return nat2exp_dataset, exp2nat_dataset, total_dataset, test_dataset

    def build_pretrain_data(self):
        """
        1. [natural_path + expression_path, natural_label + expressoin_label]
        2. shuffle 1.
        :return:
        """
        path_total = self.path_natural_train + self.path_expression_train
        label_total = self.label_natural_train + self.label_expression_train
        total = list(zip(path_total, label_total))
        random.shuffle(total)
        path_total, label_total = zip(*total)
        return list(path_total), list(label_total)

    def build_train_data(self):
        type1 = list(zip(self.path_natural_train, self.label_natural_train, self.build_ck_label(self.label_natural_train, 'expression')))
        type2 = list(zip(self.path_expression_train, self.label_expression_train, self.build_ck_label(self.label_expression_train, 'natural')))
        total = type1 + type2
        random.shuffle(total)
        ori_img_path, ori_label, tar_label = zip(*total)
        return list(ori_img_path), list(ori_label), list(tar_label)

    def build_pair_data(self):
        pairlist = []
        path1 = './classifier_alignment_CK/train/Natural image'
        path2 = './classifier_alignment_CK/train/Expression image'
        for dir in sorted(os.listdir(path1)):
            pathlist_N, pathlist_E = [], []
            path_N = path1 + '/' + dir
            path_E = path2 + '/' + dir

            for root, dirs, files in os.walk(path_N):
                for file in files:
                    if file.endswith(".png"):
                        pathlist_N.append(os.path.join(root, file))

            for root, dirs, files in os.walk(path_E):
                for file in files:
                    if file.endswith(".png"):
                        pathlist_E.append(os.path.join(root, file))

            pair = list(zip(pathlist_N, self.build_ck_label(pathlist_N, label_type='natural'), pathlist_E, self.build_ck_label(pathlist_E, label_type='expression')))
            pairlist += pair
        nat_img, nat_label, exp_img, exp_label = zip(*pairlist)
        return list(nat_img), list(nat_label), list(exp_img), list(exp_label)

    def build_oneimg_pair_data(self):
        pairlist = []
        path1 = './classifier_alignment_CK/train/Natural image'
        path2 = './classifier_alignment_CK/train/Expression image'
        for dir in sorted(os.listdir(path1)):
            pathlist_N, pathlist_E = [], []
            path_N = path1 + '/' + dir
            path_E = path2 + '/' + dir

            for root, dirs, files in os.walk(path_N):
                temp_file = []
                for file in files:
                    if file.endswith(".png"):
                        temp_file.append(os.path.join(root, file))
                if temp_file != []:
                    pathlist_N.append(temp_file[0])

            for root, dirs, files in os.walk(path_E):
                temp_file = []
                for file in files:
                    if file.endswith(".png"):
                        # print(os.path.join(root, file))
                        temp_file.append(os.path.join(root, file))
                if temp_file != []:
                    pathlist_E.append(temp_file[0])
            # print(len(path_E), len(path_N))

            pair = list(zip(pathlist_N, self.build_ck_label(pathlist_N, label_type='natural'), pathlist_E, self.build_ck_label(pathlist_E, label_type='expression')))
            print(pair)
            pairlist += pair
        nat_img, nat_label, exp_img, exp_label = zip(*pairlist)
        return list(nat_img), list(nat_label), list(exp_img), list(exp_label)


    def build_test_data(self):
        type1 = list(zip(self.path_natural_test, self.label_natural_test, self.build_ck_label(self.label_natural_test, 'expression')))
        type2 = list(zip(self.path_expression_test, self.label_expression_test, self.build_ck_label(self.label_expression_test, 'natural')))
        print(len(type1), 'here')
        total = type1 + type2
        random.shuffle(total)
        ori_img_path, ori_label, tar_label = zip(*total)
        return list(ori_img_path), list(ori_label), list(tar_label)

    def build_ck_label(self, data, label_type):
        if label_type == 'natural':
            label = [0] * (len(data))
        elif label_type == 'expression':
            label = [1] * (len(data))
        return label

    def data_dir(self, path, img_type=".png"):
        path_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(str(img_type)):
                    path_list.append(os.path.join(root, file))
        return path_list

if __name__ == '__main__':
    LData = data_loader('./classifier_alignment_CK/train/Natural image',
                        './classifier_alignment_CK/train/Expression image',
                        './classifier_alignment_CK/test/Natural image',
                        './classifier_alignment_CK/test/Expression image')
    nat2exp_dataset, exp2nat_dataset, total_dataset, test_dataset = LData.build_dataset()
