########################################
# Utilities
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# November 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
# 本模块提供 Raha benchmark 和策略过滤流程使用的工具函数。
#
# 基础使用示例：
#     import os
#     import raha
#
#     dataset_dictionary = {
#         "name": "hospital",
#         "path": os.path.abspath("datasets/hospital/dirty.csv"),
#         "clean_path": os.path.abspath("datasets/hospital/clean.csv"),
#     }
#
#     detector = raha.detection.Detection()
#     detector.run(dataset_dictionary)
#     raha.utilities.dataset_profiler(dataset_dictionary)
#     raha.utilities.evaluation_profiler(dataset_dictionary)
#     strategies_count, runtime = raha.utilities.get_strategies_count_and_runtime(dataset_dictionary)
#     print(strategies_count, runtime)
#
# 历史数据策略筛选示例：
#     historical_dataset_dictionaries = [hospital_dictionary, flights_dictionary]
#     selected = raha.utilities.get_selected_strategies_via_historical_data(
#         dataset_dictionary,
#         historical_dataset_dictionaries,
#     )
#     detected_cells = raha.utilities.error_detection_with_selected_strategies(dataset_dictionary, selected)
#
# 可行配置重点说明：
#     dataset_dictionary：
#         必须包含 name、path；需要评估或策略筛选时必须包含 clean_path。
#     historical_dataset_dictionaries：
#         历史数据集列表，需提前运行 dataset_profiler 和 evaluation_profiler。
#     strategies_count：
#         get_selected_strategies_via_ground_truth 的选取数量，只用于有真实标签的实验对照。
#     strategy_profiles_list：
#         选定策略画像列表，每项至少包含 name、output、runtime。
#     结果目录：
#         工具函数默认读写 dirty.csv 同级目录下的 raha-baran-results-<name>。
########################################


########################################
import os
import sys
import math
import json
import pickle
import random
import operator
import itertools

import scipy.spatial

import raha
########################################


########################################
def get_tuple_wise_evaluation(d, correction_dictionary):
    """
    按元组粒度评估检测或修复结果。

    输入：
        d：Dataset 对象，必须包含 clean_dataframe。
        correction_dictionary：检测或修复结果字典。

    输出：
        (precision, recall, f1)：只关心是否命中脏元组，不要求命中具体错误单元格。
    """
    actual_errors_dictionary = d.get_actual_errors_dictionary()
    # 只要某行存在至少一个真实错误单元格，该行就被视为脏元组。
    actual_dirty_tuples = {i: 1 for i in range(d.dataframe.shape[0]) if int(sum([(i, j) in actual_errors_dictionary
                           for j in range(d.dataframe.shape[1])]) > 0)}
    tp = 0.0
    outputted_tuples = {}
    for i, j in correction_dictionary:
        if i not in outputted_tuples:
            outputted_tuples[i] = 1
            if i in actual_dirty_tuples:
                # 输出元组和真实脏元组重合时，记为元组级真阳性。
                tp += 1.0
    p = 0.0 if len(outputted_tuples) == 0 else tp / len(outputted_tuples)
    r = 0.0 if len(actual_dirty_tuples) == 0 else tp / len(actual_dirty_tuples)
    f = 0.0 if (p + r) == 0.0 else (2 * p * r) / (p + r)
    return p, r, f


def dataset_profiler(dataset_dictionary):
    """
    为数据集每一列生成数据画像。

    输入：
        dataset_dictionary：数据集字典，必须包含 name 和 path。

    输出：
        无直接返回值，会写入 dataset-profiling/<列名>.dictionary。

    可行配置：
        数据画像包含字符出现比例和完整值出现比例，是历史策略筛选的相似度基础。
    """
    # 如需调试画像阶段，可临时打开下面的打印。
    # print("------------------------------------------------------------------------\n"
    #       "--------------------------Profiling the Dataset-------------------------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    dp_folder_path = os.path.join(d.results_folder, "dataset-profiling")
    if not os.path.exists(dp_folder_path):
        os.mkdir(dp_folder_path)
    for attribute in d.dataframe.columns.tolist():
        characters_dictionary = {}
        values_dictionary = {}
        for value in d.dataframe[attribute]:
            for character in list(set(list(value))):
                if character not in characters_dictionary:
                    characters_dictionary[character] = 0.0
                characters_dictionary[character] += 1.0
            if value not in values_dictionary:
                values_dictionary[value] = 0.0
            values_dictionary[value] += 1.0
        # 字符画像按“包含该字符的行占比”统计，值画像按“该完整值的行占比”统计。
        column_profile = {
            "characters": {ch: characters_dictionary[ch] / d.dataframe.shape[0] for ch in characters_dictionary},
            "values": {v: values_dictionary[v] / d.dataframe.shape[0] for v in values_dictionary},
        }
        pickle.dump(column_profile, open(os.path.join(dp_folder_path, attribute + ".dictionary"), "wb"))


def evaluation_profiler(dataset_dictionary):
    """
    计算历史数据集中每个检测策略在每列上的效果画像。

    输入：
        dataset_dictionary：必须包含 clean_path，且已存在 strategy-profiling 目录。

    输出：
        无直接返回值，会写入 evaluation-profiling/<列名>.dictionary。
    """
    # 如需调试策略评估画像阶段，可临时打开下面的打印。
    # print("------------------------------------------------------------------------\n"
    #       "---------Profiling the Performance of Strategies on the Dataset---------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    actual_errors_dictionary = d.get_actual_errors_dictionary()
    ep_folder_path = os.path.join(d.results_folder, "evaluation-profiling")
    if not os.path.exists(ep_folder_path):
        os.mkdir(ep_folder_path)
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    columns_performance = {j: {} for j in range(d.dataframe.shape[1])}
    strategies_file_list = os.listdir(sp_folder_path)
    for strategy_file in strategies_file_list:
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategy_name = strategy_profile["name"]
        strategy_output = strategy_profile["output"]
        for column_index, attribute in enumerate(d.dataframe.columns.tolist()):
            # 每个策略在每一列分别计算精确率、召回率和 F1，供历史迁移加权。
            actual_column_errors = {(i, j): 1 for (i, j) in actual_errors_dictionary if j == column_index}
            detected_column_cells = [(i, j) for (i, j) in strategy_output if j == column_index]
            tp = 0.0
            for cell in detected_column_cells:
                if cell in actual_column_errors:
                    tp += 1
            if tp == 0.0:
                precision = recall = f1 = 0.0
            else:
                precision = tp / len(detected_column_cells)
                recall = tp / len(actual_column_errors)
                f1 = (2 * precision * recall) / (precision + recall)
            columns_performance[column_index][strategy_name] = [precision, recall, f1]
    for j, attribute in enumerate(d.dataframe.columns.tolist()):
        pickle.dump(columns_performance[j], open(os.path.join(ep_folder_path, attribute + ".dictionary"), "wb"))


def get_selected_strategies_via_historical_data(dataset_dictionary, historical_dataset_dictionaries):
    """
    基于历史数据画像为当前数据集选择有希望的检测策略。

    输入：
        dataset_dictionary：当前数据集字典。
        historical_dataset_dictionaries：历史数据集字典列表。

    前置条件：
        当前数据集和历史数据集需要已生成 dataset-profiling。
        历史数据集需要已生成 evaluation-profiling。
        当前数据集需要已有完整 strategy-profiling，便于读取策略输出。

    输出：
        selected_strategy_profiles：筛选后的策略画像列表。
    """
    # 如需调试策略筛选阶段，可临时打开下面的打印。
    # print("------------------------------------------------------------------------\n"
    #       "-------Selecting Promising Strategies Based on Historical Datasets------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    columns_similarity = {}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        ndp_folder_path = os.path.join(d.results_folder, "dataset-profiling")
        ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    # 当前列和历史列使用字符分布、值分布拼接向量计算余弦相似度。
                    hdp_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-baran-results-" + hdd["name"])
                    hcp = pickle.load(open(os.path.join(hdp_folder_path, "dataset-profiling", ha + ".dictionary"), "rb"))
                    nfv = []
                    hfv = []
                    for k in list(set(ncp["characters"]) | set(hcp["characters"])):
                        nfv.append(ncp["characters"][k]) if k in ncp["characters"] else nfv.append(0.0)
                        hfv.append(hcp["characters"][k]) if k in hcp["characters"] else hfv.append(0.0)
                    for k in list(set(ncp["values"]) | set(hcp["values"])):
                        nfv.append(ncp["values"][k]) if k in ncp["values"] else nfv.append(0.0)
                        hfv.append(hcp["values"][k]) if k in hcp["values"] else hfv.append(0.0)
                    similarity = 1.0 - scipy.spatial.distance.cosine(nfv, hfv)
                    columns_similarity[(d.name, na, hd.name, ha)] = similarity
    f1_measure = {}
    for hdd in historical_dataset_dictionaries:
        if hdd["name"] != d.name:
            hd = raha.dataset.Dataset(hdd)
            for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                ep_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-baran-results-" + hdd["name"], "evaluation-profiling")
                strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                if (hd.name, ha) not in f1_measure:
                    f1_measure[(hd.name, ha)] = {}
                for strategy_name in strategies_performance:
                    # 历史列上每个策略的 F1 作为迁移评分的一部分。
                    f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
    strategies_score = {a: {} for a in d.dataframe.columns.tolist()}
    strategies_anchor = {a: {} for a in d.dataframe.columns.tolist()}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    similarity = columns_similarity[(d.name, na, hd.name, ha)]
                    anchor = [d.name, na, hd.name, ha]
                    if similarity == 0:
                        continue
                    for strategy_name in f1_measure[(hd.name, ha)]:
                        score = similarity * f1_measure[(hd.name, ha)][strategy_name]
                        if score <= 0.0:
                            continue
                        sn = json.loads(strategy_name)
                        if sn[0] == "OD" or sn[0] == "KBVD":
                            # OD 和 KBVD 策略不绑定具体列名，可直接迁移策略名称。
                            if strategy_name not in strategies_score[na] or score >= strategies_score[na][strategy_name]:
                                strategies_score[na][strategy_name] = score
                                strategies_anchor[na][strategy_name] = anchor
                        elif sn[0] == "PVD":
                            # PVD 绑定单列，迁移时替换为当前数据集的目标列名。
                            sn[1][0] = na
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                                strategies_anchor[na][json.dumps(sn)] = anchor
                        elif sn[0] == "RVD":
                            # RVD 绑定两列，先替换相似列，再为另一列寻找最相似的当前列。
                            this_a_i = sn[1].index(ha)
                            that_a = sn[1][1 - this_a_i]
                            most_similar_a = d.dataframe.columns.tolist()[0]
                            most_similar_v = -1
                            for aa in d.dataframe.columns.tolist():
                                if aa != na and columns_similarity[(d.name, aa, hd.name, that_a)] > most_similar_v:
                                    most_similar_v = columns_similarity[(d.name, aa, hd.name, that_a)]
                                    most_similar_a = aa
                            sn[1][this_a_i] = na
                            sn[1][1 - this_a_i] = most_similar_a
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                                strategies_anchor[na][json.dumps(sn)] = anchor
                        else:
                            sys.stderr.write("I do not know this error detection tool!\n")
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    strategies_output = {}
    strategies_runtime = {}
    selected_strategy_profiles = []
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategies_output[strategy_profile["name"]] = strategy_profile["output"]
        strategies_runtime[strategy_profile["name"]] = strategy_profile["runtime"]
    for a in d.dataframe.columns.tolist():
        sorted_strategies = sorted(strategies_score[a].items(), key=operator.itemgetter(1), reverse=True)
        good_strategies = {}
        previous_score = 0.0
        for sn, ss in sorted_strategies:
            if sn not in strategies_output:
                continue
            first_sum = sum(good_strategies.values())
            second_sum = sum([math.fabs(good_strategies[s_1] - good_strategies[s_2]) for s_1, s_2 in
                              itertools.product(good_strategies.keys(), good_strategies.keys()) if s_1 > s_2])
            score = first_sum - second_sum
            if score < previous_score:
                # 组合收益开始下降时停止继续加入该列策略，避免低质量策略稀释效果。
                break
            previous_score = score
            good_strategies[sn] = ss
        for sn in good_strategies:
            snd = json.loads(sn)
            runtime = 0.0
            if snd[0] == "OD" or snd[0] == "KBVD":
                runtime = strategies_runtime[sn] / d.dataframe.shape[1]
            elif snd[0] == "PVD":
                runtime = strategies_runtime[sn]
            elif snd[0] == "RVD":
                runtime = strategies_runtime[sn] / 2
            else:
                sys.stderr.write("I do not know this error detection tool!\n")
            strategy_profile = {
                "name": sn,
                # 筛选后的策略输出只保留当前列，便于后续按列生成特征。
                "output": [cell for cell in strategies_output[sn] if d.dataframe.columns.tolist()[cell[1]] == a],
                "runtime": runtime,
                "score": good_strategies[sn],
                "new_column": strategies_anchor[a][sn][0] + "." + strategies_anchor[a][sn][1],
                "historical_column": strategies_anchor[a][sn][2] + "." + strategies_anchor[a][sn][3]
            }
            selected_strategy_profiles.append(strategy_profile)
    return selected_strategy_profiles


def get_selected_strategies_via_ground_truth(dataset_dictionary, strategies_count):
    """
    使用当前数据集真实标签选择最差、随机和最佳策略集合。

    输入：
        dataset_dictionary：必须包含 clean_path，且已生成 evaluation-profiling 和 strategy-profiling。
        strategies_count：每组选择的策略数量。

    输出：
        (worst_strategy_profiles, random_strategy_profiles, best_strategy_profiles)

    注意：
        该函数依赖真实标签，只适合 benchmark 对照实验，不适合真实生产流程。
    """
    # 如需调试真实标签策略选择阶段，可临时打开下面的打印。
    # print("------------------------------------------------------------------------\n"
    #       "---Selecting Worst, Random, and Best Strategies Based on Ground Truth---\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    f1_measure = {}
    ep_folder_path = os.path.join(d.results_folder, "evaluation-profiling")
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        strategies_performance = pickle.load(open(os.path.join(ep_folder_path, na + ".dictionary"), "rb"))
        for strategy_name in strategies_performance:
            f1_measure[(na, strategy_name)] = strategies_performance[strategy_name][2]
    sorted_f1_measure = sorted(f1_measure.items(), key=operator.itemgetter(1))
    # 按 F1 排序后取最差、随机和最佳三组策略，用于实验中的上界和下界对比。
    worst_strategies = {s: f1 for s, f1 in sorted_f1_measure[:strategies_count]}
    random_strategies = {s: f1 for s, f1 in [sorted_f1_measure[i] for i in
                                             random.sample(range(len(sorted_f1_measure)), strategies_count)]}
    best_strategies = {s: f1 for s, f1 in sorted_f1_measure[-strategies_count:]}
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    worst_strategy_profiles = []
    random_strategy_profiles = []
    best_strategy_profiles = []
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        for a in d.dataframe.columns.tolist():
            snd = json.loads(strategy_profile["name"])
            runtime = 0.0
            if snd[0] == "OD" or snd[0] == "KBVD":
                runtime = strategy_profile["runtime"] / d.dataframe.shape[1]
            elif snd[0] == "PVD":
                runtime = strategy_profile["runtime"]
            elif snd[0] == "RVD":
                runtime = strategy_profile["runtime"] / 2
            else:
                sys.stderr.write("I do not know this error detection tool!\n")
            sp = {
                "name": strategy_profile["name"],
                # 每个策略画像按列拆分，便于和列级 F1 对齐。
                "output": [cell for cell in strategy_profile["output"] if d.dataframe.columns.tolist()[cell[1]] == a],
                "runtime": runtime,
                "score": f1_measure[(a, strategy_profile["name"])]
            }
            if (a, strategy_profile["name"]) in worst_strategies:
                worst_strategy_profiles.append(sp)
            if (a, strategy_profile["name"]) in random_strategies:
                random_strategy_profiles.append(sp)
            if (a, strategy_profile["name"]) in best_strategies:
                best_strategy_profiles.append(sp)
    return worst_strategy_profiles, random_strategy_profiles, best_strategy_profiles


def get_strategies_count_and_runtime(dataset_dictionary):
    """
    统计完整策略画像中的策略数量和总运行时间。

    输入：
        dataset_dictionary：已生成 strategy-profiling 的数据集字典。

    输出：
        (strategies_count, strategies_runtime)
    """
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    strategies_count = 0
    strategies_runtime = 0
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategies_runtime += strategy_profile["runtime"]
        sn = json.loads(strategy_profile["name"])
        if sn[0] in ["OD", "KBVD"]:
            # OD 和 KBVD 的一次策略运行可产生多列输出，统计时按列数折算。
            strategies_count += d.dataframe.shape[1]
        if sn[0] in ["PVD", "RVD"]:
            strategies_count += 1
    return strategies_count, strategies_runtime


def error_detection_with_selected_strategies(dataset_dictionary, strategy_profiles_list):
    """
    仅使用给定策略画像运行 Raha 后半段检测流程。

    输入：
        dataset_dictionary：数据集字典。
        strategy_profiles_list：选定策略画像列表。

    输出：
        detected_cells：错误单元格字典。

    可行配置：
        strategy_profiles_list 可来自历史数据筛选、真实标签筛选或外部自定义策略选择。
    """
    app = raha.Detection()
    # 如需调试完整流程，可临时打开各阶段打印。
    # print("------------------------------------------------------------------------\n"
    #       "--------------------Instantiating the Dataset Object--------------------\n"
    #       "------------------------------------------------------------------------")
    d = app.initialize_dataset(dataset_dictionary)
    # 跳过 run_strategies，直接注入外部筛选好的策略画像。
    # print("------------------------------------------------------------------------\n"
    #       "-------------------Running Error Detection Strategies-------------------\n"
    #       "------------------------------------------------------------------------")
    d.strategy_profiles = strategy_profiles_list
    # print("------------------------------------------------------------------------\n"
    #       "-----------------------Generating Feature Vectors-----------------------\n"
    #       "------------------------------------------------------------------------")
    app.generate_features(d)
    # print("------------------------------------------------------------------------\n"
    #       "---------------Building the Hierarchical Clustering Model---------------\n"
    #       "------------------------------------------------------------------------")
    app.build_clusters(d)
    # print("------------------------------------------------------------------------\n"
    #       "-------------Iterative Clustering-Based Sampling and Labeling-----------\n"
    #       "------------------------------------------------------------------------")
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d)
        if d.has_ground_truth:
            # benchmark 场景有 clean_path 时自动标注。
            app.label_with_ground_truth(d)
    # print("------------------------------------------------------------------------\n"
    #       "--------------Propagating User Labels Through the Clusters--------------\n"
    #       "------------------------------------------------------------------------")
    app.propagate_labels(d)
    # print("------------------------------------------------------------------------\n"
    #       "---------------Training and Testing Classification Models---------------\n"
    #       "------------------------------------------------------------------------")
    app.predict_labels(d)
    # if app.SAVE_RESULTS:
    #     print("------------------------------------------------------------------------\n"
    #           "---------------------------Storing the Results--------------------------\n"
    #           "------------------------------------------------------------------------")
    #     app.store_results(d)
    return d.detected_cells
