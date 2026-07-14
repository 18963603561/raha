########################################
# Baselines
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
# 本模块封装论文实验中使用的对比基线方法。
#
# 基础使用示例：
#     import os
#     from raha.baselines import Baselines
#
#     dataset_dictionary = {
#         "name": "hospital",
#         "path": os.path.abspath("datasets/hospital/dirty.csv"),
#         "clean_path": os.path.abspath("datasets/hospital/clean.csv"),
#     }
#
#     app = Baselines()
#     app.VERBOSE = True
#     detection_dictionary = app.run_nadeef(dataset_dictionary)
#     print(detection_dictionary)
#
# 可行配置重点说明：
#     VERBOSE：
#         是否输出运行阶段信息。
#     DATASET_CONSTRAINTS：
#         数据集到人工规则的映射，仅 NADEEF 和 Metadata Driven 直接依赖。
#         functions 表示函数依赖规则 [左列, 右列]。
#         patterns 表示模式规则 [列名, 正则表达式, 操作码]。
#         操作码 "OM" 表示匹配即异常，"ONM" 表示不匹配即异常。
#     sampling_budget：
#         run_activeclean、run_maximum_entropy、run_metadata_driven 的采样预算。
#         值越大可用真实标签越多，通常效果更稳但成本更高。
#
# 注意：
#     dBoost、KATARA、Min-k、Maximum Entropy 依赖 Raha 事先生成的 strategy-profiling 缓存。
#     可先运行 Detection().run(dataset_dictionary) 生成对应目录。
########################################


########################################
import os
import re
import json
import random
import pickle
import operator
import itertools

import numpy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.feature_extraction

import raha
########################################


########################################
class Baselines:
    """
    对比基线方法集合。

    各方法返回 detection_dictionary，键为 (row_index, column_index)，值为占位字符串。
    """

    def __init__(self):
        """
        初始化基线配置。

        可重点调整 DATASET_CONSTRAINTS，以适配新增数据集的函数依赖和模式规则。
        """
        # 是否输出运行过程信息。
        self.VERBOSE = False
        # 内置数据集约束规则，供 NADEEF 和 Metadata Driven 使用。
        self.DATASET_CONSTRAINTS = {
            "hospital": {
                "functions": [["city", "zip"], ["city", "county"], ["zip", "city"], ["zip", "state"], ["zip", "county"],
                              ["county", "state"]],
                "patterns": [["index", r"^[\d]+$", "ONM"], ["provider_number", r"^[\d]+$", "ONM"],
                             ["zip", r"^[\d]{5}$", "ONM"], ["state", "^[a-z]{2}$", "ONM"], ["phone", r"^[\d]+$", "ONM"]]
            },
            "flights": {
                "functions": [["flight", "act_dep_time"], ["flight", "sched_arr_time"], ["flight", "act_arr_time"],
                              ["flight", "sched_dep_time"]],
                "patterns": []
            },
            "address": {
                "functions": [["address", "state"], ["address", "zip"], ["zip", "state"]],
                "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["zip", r"^[\d]+$", "ONM"], ["ssn", r"^[\d]*$", "ONM"]]
            },
            "beers": {
                "functions": [["brewery_id", "brewery_name"], ["brewery_id", "city"], ["brewery_id", "state"]],
                "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["brewery_id", r"^[\d]+$", "ONM"]]
            },
            "rayyan": {
                "functions": [["jounral_abbreviation", "journal_title"], ["jounral_abbreviation", "journal_issn"],
                              ["journal_issn", "journal_title"]],
                "patterns": [
                    ["article_jvolumn", "^$", "OM"],
                    ["article_jissue", "^$", "OM"],
                    ["article_jcreated_at", r"^[\d]+[/][\d]+[/][\d]+$|^$", "OM"],
                    ["journal_issn", "^$", "OM"],
                    ["journal_title", "^$", "OM"],
                    ["article_language", "^$", "OM"],
                    ["article_title", "^$", "OM"],
                    ["jounral_abbreviation", "^$", "OM"],
                    ["article_pagination", "^$", "OM"],
                    ["author_list", "^$", "OM"],
                    # ["journal_issn", "^[A-Z][a-z]{2}[-][012][\d]$", "OM"],
                    # ["article_title", """^[A-Za-z_\d [\]<>!?:;./,*()+&'"%-]+$""", "ONM"],
                    # ["journal_title", """^[A-Za-z_\d [\]=:;./,()&'-]+$|^$""", "ONM"],
                    # ["author_list", """^[A-Za-z_\d [\]:./,}{@()&'-]+$""", "ONM"]
                ]
            },
            "movies_1": {
                "functions": [],
                "patterns": [["id", r"^tt[\d]+$", "ONM"], ["year", r"^[\d]{4}$", "ONM"],
                             ["rating_value", r"^[\d.]*$", "ONM"],
                             # ["rating_count", "^[\d]*$", "ONM"],
                             ["duration", r"^([\d]+[ ]min)*$", "ONM"]]
            },
            "merck": {
                "functions": [],
                "patterns": [["support_level", "^$", "OM"], ["app_status", "^$", "OM"], ["curr_status", "^$", "OM"],
                             ["tower", "^$", "OM"], ["end_users", "^$", "OM"], ["account_manager", "^$", "OM"],
                             ["decomm_dt", "^$", "OM"], ["decomm_start", "^$", "OM"], ["decomm_end", "^$", "OM"],
                             ["end_users", "^(0)$", "OM"],
                             ["retirement", "^(2010|2011|2012|2013|2014|2015|2016|2017|2018)$", "ONM"],
                             ["emp_dta", "^(n|N|y|Y|n/a|N/A|n/A|N/a)$", "ONM"],
                             ["retire_plan", "^(true|True|TRUE|false|False|FALSE|n/a|N/A|n/A|N/a)$", "ONM"],
                             ["bus_import", "^(important|n/a|IP Strategy)$", "OM"],
                             ["division", "^(Merck Research Laboratories|Merck Consumer Health Care)$", "OM"]]
            }
        }

    def run_dboost(self, dd):
        """
        运行 dBoost 基线。

        输入：
            dd：数据集字典，必须包含 clean_path 以便在采样行上选择最佳 OD 策略。

        前置条件：
            需要已存在 strategy-profiling 目录，通常由 Detection().run(dd) 生成。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------------------运行 dBoost-----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        # 随机抽取 1% 元组作为验证样本，用于从多个 OD 策略中选择 F1 最好的策略。
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:int(d.dataframe.shape[0] / 100.0)]}
        best_f1 = -1.0
        best_strategy = ""
        detection_dictionary = {}
        for strategy_profile in strategy_profiles_list:
            algorithm = json.loads(strategy_profile["name"])[0]
            if algorithm == "OD":
                strategy_output = {cell: "JUST A DUUMY VALUE" for cell in strategy_profile["output"]}
                er = d.get_data_cleaning_evaluation(strategy_output, sampled_rows_dictionary=labeled_tuples)[:3]
                if er[2] > best_f1:
                    # 只保留采样验证集上 F1 最优的异常检测策略输出。
                    best_f1 = er[2]
                    best_strategy = strategy_profile["name"]
                    detection_dictionary = dict(strategy_output)
        return detection_dictionary

    def run_nadeef(self, dd):
        """
        运行基于人工约束的 NADEEF 风格基线。

        可行配置：
            DATASET_CONSTRAINTS[数据集名]["functions"] 控制函数依赖规则。
            DATASET_CONSTRAINTS[数据集名]["patterns"] 控制正则模式规则。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------------------运行 NADEEF-----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        detection_dictionary = {}
        for fd in self.DATASET_CONSTRAINTS[d.name]["functions"]:
            l_attribute, r_attribute = fd
            l_j = d.dataframe.columns.get_loc(l_attribute)
            r_j = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            for i, row in d.dataframe.iterrows():
                if row[l_attribute]:
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    if row[r_attribute]:
                        value_dictionary[row[l_attribute]][row[r_attribute]] = 1
            for i, row in d.dataframe.iterrows():
                if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
                    # 左列同一取值映射到多个右列取值时，认为相关单元格违反函数依赖。
                    detection_dictionary[(i, l_j)] = "JUST A DUUMY VALUE"
                    detection_dictionary[(i, r_j)] = "JUST A DUUMY VALUE"
        for attribute, pattern, opcode in self.DATASET_CONSTRAINTS[d.name]["patterns"]:
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].iteritems():
                if opcode == "OM":
                    # OM 表示匹配该模式即异常。
                    if len(re.findall(pattern, value, re.UNICODE)) > 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                else:
                    # ONM 表示不匹配该模式即异常。
                    if len(re.findall(pattern, value, re.UNICODE)) == 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
        return detection_dictionary

    def run_katara(self, dd):
        """
        运行 KATARA 知识库检测基线。

        前置条件：
            需要 strategy-profiling 中已有 KBVD 策略输出。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------------------运行 KATARA-----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        detection_dictionary = {}
        for strategy_profile in strategy_profiles_list:
            algorithm = json.loads(strategy_profile["name"])[0]
            if algorithm == "KBVD":
                # KATARA 对应 Raha 内部的 KBVD 策略，多个知识库输出取并集。
                detection_dictionary.update({cell: "JUST A DUUMY VALUE" for cell in strategy_profile["output"]})
        return detection_dictionary

    def run_activeclean(self, dd, sampling_budget=20):
        """
        运行 ActiveClean 风格的主动学习基线。

        可行配置：
            sampling_budget：标注元组数量，默认 20。

        输出：
            被预测为脏的元组会展开为整行所有单元格。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "----------------------------运行 ActiveClean---------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
        text = [" ".join(row) for row in d.dataframe.values.tolist()]
        acfv = vectorizer.fit_transform(text).toarray()
        labeled_tuples = {}
        adaptive_detector_output = []
        detection_dictionary = {}
        while len(labeled_tuples) < sampling_budget:
            if len(adaptive_detector_output) < 1:
                # 初始阶段或没有候选输出时，从所有未标注行中继续抽样。
                adaptive_detector_output = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
            labeled_tuples.update({i: 1 for i in numpy.random.choice(adaptive_detector_output, 1, replace=False)})
            x_train = []
            y_train = []
            for i in labeled_tuples:
                x_train.append(acfv[i, :])
                y_train.append(int(sum([(i, j) in actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0))
            adaptive_detector_output = []
            x_test = [acfv[i, :] for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
            test_rows = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
            if sum(y_train) == len(y_train):
                # 训练样本全为脏元组时，剩余行全部预测为脏。
                predicted_labels = len(test_rows) * [1]
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                # 没有脏样本或没有有效特征时，剩余行全部预测为干净。
                predicted_labels = len(test_rows) * [0]
            else:
                # 使用文本 TF-IDF 特征训练线性分类器，预测元组是否包含错误。
                model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True)
                model.fit(x_train, y_train)
                predicted_labels = model.predict(x_test)
            detection_dictionary = {}
            for index, pl in enumerate(predicted_labels):
                i = test_rows[index]
                if pl:
                    adaptive_detector_output.append(i)
                    for j in range(d.dataframe.shape[1]):
                        detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
            for i in labeled_tuples:
                for j in range(d.dataframe.shape[1]):
                    detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
        return detection_dictionary

    def run_min_k(self, dd):
        """
        运行 Min-k 聚合基线。

        前置条件：
            需要已存在 strategy-profiling 目录。

        可行配置：
            thresholds_list 是内部候选阈值列表，可按实验需要扩展。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------------------运行 Min-k------------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        cells_counter = {}
        for strategy_profile in strategy_profiles_list:
            for cell in strategy_profile["output"]:
                if cell not in cells_counter:
                    cells_counter[cell] = 0.0
                cells_counter[cell] += 1.0
        for cell in cells_counter:
            # 每个单元格的分数是命中该单元格的策略占比。
            cells_counter[cell] /= len(strategy_profiles_list)
        thresholds_list = [0.0, 0.2, 0.4, 0.6, 0.8]
        detection_dictionary = {}
        best_f1 = 0.0
        for k in thresholds_list:
            temp_output = {}
            for cell in cells_counter:
                if cells_counter[cell] >= k:
                    temp_output[cell] = "JUST A DUMMY VALUE"
            er = d.get_data_cleaning_evaluation(temp_output)[:3]
            if er[2] > best_f1:
                # 使用 clean_path 评估不同阈值，保留 F1 最好的聚合结果。
                best_f1 = er[2]
                detection_dictionary = dict(temp_output)
        return detection_dictionary

    def run_maximum_entropy(self, dd, sampling_budget=20):
        """
        运行 Maximum Entropy 策略选择基线。

        可行配置：
            sampling_budget：通过策略输出累积标注的目标元组数量。

        前置条件：
            需要 clean_path 和 strategy-profiling。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------------运行 Maximum Entropy-------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:10]}
        detection_dictionary = {}
        while len(labeled_tuples) < sampling_budget:
            best_precision = -1.0
            best_strategy_index = 0
            for strategy_index, strategy_profile in enumerate(list(strategy_profiles_list)):
                tp = 0.0
                for cell in strategy_profile["output"]:
                    if cell in actual_errors_dictionary:
                        tp += 1
                precision = 0.0 if len(strategy_profile["output"]) == 0 else tp / len(strategy_profile["output"])
                if precision > best_precision:
                    # 每轮选择全局精确率最高的剩余策略。
                    best_precision = precision
                    best_strategy_index = strategy_index
            for cell in strategy_profiles_list[best_strategy_index]["output"]:
                detection_dictionary[cell] = "JUST A DUMMY VALUE"
                labeled_tuples[cell[0]] = 1
            strategy_profiles_list.pop(best_strategy_index)
        return detection_dictionary

    def run_metadata_driven(self, dd, sampling_budget=20):
        """
        运行 Metadata Driven 聚合基线。

        可行配置：
            sampling_budget：随机标注元组数量。
            DATASET_CONSTRAINTS：约束特征来源。

        输出：
            使用多类元特征训练 AdaBoost，预测单元格是否为错误。
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------------运行 Metadata Driven-------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        dboost_output = self.run_dboost(dd)
        nadeef_output = self.run_nadeef(dd)
        katara_output = self.run_katara(dd)
        lfv = {}
        columns_frequent_values = {}
        for j, attribute in enumerate(d.dataframe.columns.tolist()):
            fd = {}
            for value in d.dataframe[attribute].tolist():
                if value not in fd:
                    fd[value] = 0
                fd[value] += 1
            sorted_fd = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)[:int(d.dataframe.shape[0] / 10.0)]
            columns_frequent_values[j] = {v: f for v, f in sorted_fd}
        cells_list = list(itertools.product(range(d.dataframe.shape[0]), range(d.dataframe.shape[1])))
        for cell in cells_list:
            lfv[cell] = []
            # 前三维元特征来自 dBoost、NADEEF、KATARA 是否命中当前单元格。
            lfv[cell] += [1 if cell in dboost_output else 0]
            lfv[cell] += [1 if cell in nadeef_output else 0]
            lfv[cell] += [1 if cell in katara_output else 0]
            value = d.dataframe.iloc[cell[0], cell[1]]
            # 后续元特征描述常见值、IP、URL、数字、邮箱、性别、空值和约束相关列等模式。
            lfv[cell] += [1 if value in columns_frequent_values[cell[1]] else 0]
            lfv[cell] += [1 if re.findall(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value) else 0]
            lfv[cell] += [1 if re.findall(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", value) else 0]
            lfv[cell] += [1 if re.findall(r"^[\d]+$", value) else 0]
            lfv[cell] += [1 if re.findall(r"[\w.-]+@[\w.-]+", value) else 0]
            lfv[cell] += [1 if re.findall(r"^[\d]{16}$", value) else 0]
            lfv[cell] += [1 if value.lower() in ["m", "f"] else 0]
            lfv[cell] += [1 if re.findall(r"^[\d]{4,6}$", value) else 0]
            lfv[cell] += [1 if not value else 0]
            for la, ra in self.DATASET_CONSTRAINTS[d.name]["functions"]:
                lfv[cell] += [1 if d.dataframe.columns.tolist()[cell[1]] in [la, ra] else 0]
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:sampling_budget]}
        x_train = []
        y_train = []
        for cell in cells_list:
            if cell[0] in labeled_tuples:
                x_train.append(lfv[cell])
                y_train.append(int(cell in actual_errors_dictionary))
        detection_dictionary = {}
        if sum(y_train) != 0:
            x_test = [lfv[cell] for cell in cells_list]
            test_cells = [cell for cell in cells_list]
            if sum(y_train) != len(y_train):
                # 有正负样本时训练 AdaBoost；全正样本时直接全部预测为错误。
                model = sklearn.ensemble.AdaBoostClassifier(n_estimators=6)
                model.fit(x_train, y_train)
                predicted_labels = model.predict(x_test)
            else:
                predicted_labels = len(test_cells) * [1]
            detection_dictionary = {}
            for index, pl in enumerate(predicted_labels):
                cell = test_cells[index]
                if cell[0] in labeled_tuples:
                    if cell in actual_errors_dictionary:
                        detection_dictionary[cell] = "JUST A DUMMY VALUE"
                elif pl:
                    detection_dictionary[cell] = "JUST A DUMMY VALUE"
        return detection_dictionary
########################################


########################################
if __name__ == "__main__":
    # 直接运行本文件时，演示 hospital 数据集上的 dBoost 基线调用。
    dataset_name = "hospital"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = Baselines()
    app.run_dboost(dataset_dictionary)
########################################
