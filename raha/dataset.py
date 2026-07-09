########################################
# Dataset
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
# 本模块封装 Raha / Baran 使用的数据集读写、差异比较和评估逻辑。
#
# 基础使用示例：
#     import os
#     from raha.dataset import Dataset
#
#     dataset_dictionary = {
#         "name": "flights",
#         "path": os.path.abspath("datasets/flights/dirty.csv"),
#         "clean_path": os.path.abspath("datasets/flights/clean.csv"),
#         "repaired_path": os.path.abspath("datasets/flights/repaired.csv"),
#     }
#
#     data = Dataset(dataset_dictionary)
#     actual_errors = data.get_actual_errors_dictionary()
#     quality = data.get_data_quality()
#     print(actual_errors, quality)
#
# 可行配置重点说明：
#     name：
#         数据集名称，用于结果目录命名和部分基线规则匹配；必须提供。
#     path：
#         脏数据 CSV 路径；必须提供，CSV 需为 UTF-8 编码并带表头。
#     clean_path：
#         干净数据 CSV 路径；可选。提供后可计算真实错误、自动标注和评估指标。
#     repaired_path：
#         已修复数据 CSV 路径；可选。提供后可比较修复前后差异。
#     correction_dictionary：
#         修复结果字典，键为 (row_index, column_index)，值为修复后的字符串。
########################################


########################################
import re
import sys
import html

import pandas
########################################


########################################
class Dataset:
    """
    数据集对象。

    该类负责读取 CSV、规范化单元格值、比较脏数据与干净数据差异，并计算检测和修复指标。
    """

    def __init__(self, dataset_dictionary):
        """
        根据数据集字典创建 Dataset。

        输入：
            dataset_dictionary：包含 name、path，可选 clean_path、repaired_path 的字典。

        注意：
            path、clean_path、repaired_path 对应 CSV 的形状应保持一致，否则差异比较结果不可靠。
        """
        # 数据集名称，后续结果目录和基线规则会使用该名称。
        self.name = dataset_dictionary["name"]
        # 脏数据 CSV 路径，是 Detection 和 Correction 的主要输入。
        self.path = dataset_dictionary["path"]
        # 脏数据 dataframe，所有单元格按字符串读取并做最小规范化。
        self.dataframe = self.read_csv_dataset(dataset_dictionary["path"])
        if "clean_path" in dataset_dictionary:
            # clean_path 存在时，表示当前数据集可用于自动标注和离线评估。
            self.has_ground_truth = True
            self.clean_path = dataset_dictionary["clean_path"]
            self.clean_dataframe = self.read_csv_dataset(dataset_dictionary["clean_path"])
        if "repaired_path" in dataset_dictionary:
            # repaired_path 存在时，表示外部已经生成了修复结果，可用于对比修复字典。
            self.has_been_repaired = True
            self.repaired_path = dataset_dictionary["repaired_path"]
            self.repaired_dataframe = self.read_csv_dataset(dataset_dictionary["repaired_path"])

    @staticmethod
    def value_normalizer(value):
        """
        对单元格值做最小规范化。

        输入：
            value：原始字符串值。

        输出：
            规范化后的字符串，主要处理 HTML 转义、连续空白和首尾空白。
        """
        # CSV 读取后统一解除 HTML 转义，避免实体编码影响差异比较。
        value = html.unescape(value)
        # 将制表符、换行和连续空格合并为一个空格。
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value

    def read_csv_dataset(self, dataset_path):
        """
        从 CSV 文件读取数据集。

        可行配置：
            CSV 固定按逗号分隔、UTF-8 编码、首行表头读取，空值保留为空字符串。
        """
        # dtype=str 和 keep_default_na=False 保证所有值按字符串处理，避免数值和空值被 pandas 自动转换。
        dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).map(self.value_normalizer)
        return dataframe

    @staticmethod
    def write_csv_dataset(dataset_path, dataframe):
        """
        将 dataframe 写入 CSV 文件。

        输出 CSV 固定使用 UTF-8 编码、逗号分隔、保留表头且不写入索引。
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

    @staticmethod
    def get_dataframes_difference(dataframe_1, dataframe_2):
        """
        比较两个 dataframe，并返回不同单元格。

        输出：
            difference_dictionary：键为 (row_index, column_index)，值为 dataframe_2 中的目标值。
        """
        if dataframe_1.shape != dataframe_2.shape:
            # 形状不一致时继续尝试比较，但会向 stderr 输出告警。
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        difference_dictionary = {}
        # dataframe_1 与 dataframe_2 值不同的位置被视为差异单元格。
        difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
        for j in range(dataframe_1.shape[1]):
            for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
                difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
        return difference_dictionary

    def create_repaired_dataset(self, correction_dictionary):
        """
        根据修复字典生成 repaired_dataframe。

        输入：
            correction_dictionary：键为 (row_index, column_index)，值为修复后的字符串。
        """
        self.repaired_dataframe = self.dataframe.copy()
        for cell in correction_dictionary:
            # 写入修复值前复用最小规范化逻辑，保持和 CSV 读取一致。
            self.repaired_dataframe.iloc[cell] = self.value_normalizer(correction_dictionary[cell])

    def get_actual_errors_dictionary(self):
        """
        比较脏数据和干净数据，返回真实错误字典。

        前提：
            dataset_dictionary 必须提供 clean_path。
        """
        return self.get_dataframes_difference(self.dataframe, self.clean_dataframe)

    def get_correction_dictionary(self):
        """
        比较修复后数据和脏数据，返回修复变更字典。

        前提：
            已提供 repaired_path，或先调用 create_repaired_dataset。
        """
        return self.get_dataframes_difference(self.dataframe, self.repaired_dataframe)

    def get_data_quality(self):
        """
        计算数据质量。

        输出：
            1 - 真实错误单元格数 / 总单元格数。
        """
        return 1.0 - float(len(self.get_actual_errors_dictionary())) / (self.dataframe.shape[0] * self.dataframe.shape[1])

    def get_data_cleaning_evaluation(self, correction_dictionary, sampled_rows_dictionary=False):
        """
        评估错误检测或错误修复结果。

        输入：
            correction_dictionary：检测或修复结果字典。
            sampled_rows_dictionary：可选，仅在指定行集合上评估。

        输出：
            [ed_p, ed_r, ed_f, ec_p, ec_r, ec_f]
            分别表示错误检测精确率、召回率、F1，以及错误修复精确率、召回率、F1。
        """
        actual_errors = self.get_actual_errors_dictionary()
        if sampled_rows_dictionary:
            # 只评估采样行时，真实错误集合也限制在这些行内。
            actual_errors = {(i, j): actual_errors[(i, j)] for (i, j) in actual_errors if i in sampled_rows_dictionary}
        ed_tp = 0.0
        ec_tp = 0.0
        output_size = 0.0
        for cell in correction_dictionary:
            if (not sampled_rows_dictionary) or (cell[0] in sampled_rows_dictionary):
                output_size += 1
                if cell in actual_errors:
                    # 检测命中真实错误记为检测真阳性。
                    ed_tp += 1.0
                    if correction_dictionary[cell] == actual_errors[cell]:
                        # 修复值也等于干净数据时，记为修复真阳性。
                        ec_tp += 1.0
        ed_p = 0.0 if output_size == 0 else ed_tp / output_size
        ed_r = 0.0 if len(actual_errors) == 0 else ed_tp / len(actual_errors)
        ed_f = 0.0 if (ed_p + ed_r) == 0.0 else (2 * ed_p * ed_r) / (ed_p + ed_r)
        ec_p = 0.0 if output_size == 0 else ec_tp / output_size
        ec_r = 0.0 if len(actual_errors) == 0 else ec_tp / len(actual_errors)
        ec_f = 0.0 if (ec_p + ec_r) == 0.0 else (2 * ec_p * ec_r) / (ec_p + ec_r)
        return [ed_p, ed_r, ed_f, ec_p, ec_r, ec_f]
########################################


########################################
if __name__ == "__main__":
    # 直接运行本文件时，演示如何读取 toy 数据集并计算数据质量。
    dataset_dict = {
        "name": "toy",
        "path": "datasets/dirty.csv",
        "clean_path": "datasets/clean.csv"
    }
    d = Dataset(dataset_dict)
    print(d.get_data_quality())
########################################
