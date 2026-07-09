########################################
# Raha: The Error Detection System
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
# 本模块实现 Raha 的错误检测主流程。
#
# 基础使用示例：
#     import os
#     from raha.detection import Detection
#
#     dataset_name = "flights"
#     dataset_dictionary = {
#         "name": dataset_name,
#         "path": os.path.abspath("datasets/flights/dirty.csv"),
#         "clean_path": os.path.abspath("datasets/flights/clean.csv"),
#     }
#
#     app = Detection()
#     app.LABELING_BUDGET = 20
#     app.VERBOSE = True
#     detected_cells = app.run(dataset_dictionary)
#     print(detected_cells)
#
# 交互式标注场景示例：
#     dataset_dictionary 中可以只提供 dirty.csv 路径；
#     当没有 clean_path 或数据集没有真实标签时，需要在 notebook 中人工标注抽样元组。
#
# 历史数据策略过滤示例：
#     app = Detection()
#     app.STRATEGY_FILTERING = True
#     app.HISTORICAL_DATASETS = [
#         {
#             "name": "hospital",
#             "path": "/path/to/hospital/dirty.csv",
#             "clean_path": "/path/to/hospital/clean.csv",
#         }
#     ]
#     detected_cells = app.run(dataset_dictionary)
########################################


########################################
import os
import re
import sys
import math
import time
import json
import random
import pickle
import hashlib
import tempfile
import itertools
import multiprocessing

import numpy
import pandas
import scipy.stats
import scipy.spatial
import scipy.cluster
import sklearn.svm
import sklearn.tree
import sklearn.cluster
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.kernel_ridge
import sklearn.neural_network
import sklearn.feature_extraction

import raha
########################################


########################################
class Detection:
    """
    Raha 错误检测主类。

    该类把多个基础错误检测策略的输出转换为特征，随后通过抽样标注、标签传播和分类模型，
    预测整张表中可能存在错误的数据单元格。

    输入：
        dd：数据集字典，通常包含 name、path，可选 clean_path。

    输出：
        run 方法返回 detected_cells 字典，键为 (row_index, column_index)，值为占位内容。

    注意：
        当 clean_path 存在时，流程可自动使用真实标签；否则需要外部交互流程补充用户标注。
    """

    def __init__(self):
        """
        初始化错误检测流程的默认配置。

        默认配置偏向论文实验场景：运行全部基础检测策略，使用聚类抽样，并保存中间结果。
        """
        # 用户最多标注的元组数量，预算越大通常召回越高，但人工成本也越高。
        self.LABELING_BUDGET = 20
        # 模拟用户标注准确率；1.0 表示完全相信真实标签，低于 1.0 会随机翻转部分标注。
        self.USER_LABELING_ACCURACY = 1.0
        # 是否输出详细过程信息，适合调试和理解流水线。
        self.VERBOSE = False
        # 是否把策略画像和检测结果保存到数据集旁边的结果目录。
        self.SAVE_RESULTS = True
        # 是否基于聚类结果选择下一条待标注元组；关闭后退化为随机抽样。
        self.CLUSTERING_BASED_SAMPLING = True
        # 是否使用历史数据筛选更有希望的检测策略。
        self.STRATEGY_FILTERING = False
        # 最终集成阶段使用的分类模型，可选值包括 ABC、DTC、GBC、GNB、SGDC、SVC。
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        # 标签传播方式：homogeneity 要求簇内已标注样本一致，majority 使用多数投票。
        self.LABEL_PROPAGATION_METHOD = "homogeneity"   # ["homogeneity", "majority"]
        # 基础错误检测策略集合：OD 异常值、PVD 模式违规、RVD 规则违规、KBVD 知识库违规。
        self.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]   # ["OD", "PVD", "RVD", "KBVD", "TFIDF"]
        # 历史数据集配置列表，仅在 STRATEGY_FILTERING 开启时参与策略筛选。
        self.HISTORICAL_DATASETS = []

    def _strategy_runner_process(self, args):
        """
        在子进程中运行单个基础错误检测策略。

        输入：
            args：三元组，包含数据集对象、算法名称和算法配置。

        输出：
            strategy_profile：策略画像，包含策略名称、输出单元格和运行耗时。

        注意：
            该方法会被 multiprocessing.Pool 调用，因此入参必须可序列化。
        """
        d, algorithm, configuration = args
        start_time = time.time()
        strategy_name = json.dumps([algorithm, configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        outputted_cells = {}
        if algorithm == "OD":
            # OD 依赖 dBoost，需要先把当前数据集写入临时 CSV 文件供外部工具读取。
            dataset_path = os.path.join(tempfile.gettempdir(), d.name + "-" + strategy_name_hash + ".csv")
            d.write_csv_dataset(dataset_path, d.dataframe)
            params = ["-F", ",", "--statistical", "0.5"] + ["--" + configuration[0]] + configuration[1:] + [dataset_path]
            raha.tools.dBoost.dboost.imported_dboost.run(params)
            algorithm_results_path = dataset_path + "-dboost_output.csv"
            if os.path.exists(algorithm_results_path):
                # dBoost 输出的行号包含表头偏移，这里把行号转换回 dataframe 的零基索引。
                ocdf = pandas.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
                os.remove(algorithm_results_path)
            os.remove(dataset_path)
        elif algorithm == "PVD":
            # PVD 针对单列中特定字符的出现模式生成候选错误单元格。
            attribute, ch = configuration
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].items():
                try:
                    if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                        outputted_cells[(i, j)] = ""
                except:
                    # 某些字符会导致正则表达式无法解析，跳过该值以保证策略继续运行。
                    continue
        elif algorithm == "RVD":
            # RVD 查找左列到右列的一对多映射，出现冲突时两列对应单元格都作为候选错误。
            l_attribute, r_attribute = configuration
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
                    outputted_cells[(i, l_j)] = ""
                    outputted_cells[(i, r_j)] = ""
        elif algorithm == "KBVD":
            # KBVD 调用 KATARA 知识库检测器，configuration 是具体知识库文件路径。
            outputted_cells = raha.tools.KATARA.katara.run(d, configuration)
        detected_cells_list = list(outputted_cells.keys())
        strategy_profile = {
            "name": strategy_name,
            "output": detected_cells_list,
            "runtime": time.time() - start_time
        }
        if self.SAVE_RESULTS:
            # 策略画像会被缓存，下次同一数据集可直接加载以节省运行时间。
            pickle.dump(strategy_profile, open(os.path.join(d.results_folder, "strategy-profiling",
                                                            strategy_name_hash + ".dictionary"), "wb"))
        if self.VERBOSE:
            print("{} cells are detected by {}.".format(len(detected_cells_list), strategy_name))
        return strategy_profile

    def initialize_dataset(self, dd):
        """
        根据数据集字典创建并初始化 Dataset 对象。

        输入：
            dd：数据集字典，必须包含 name 和 path，可选 clean_path。

        输出：
            d：带有结果目录和标注状态容器的 Dataset 对象。
        """
        d = raha.dataset.Dataset(dd)
        d.dictionary = dd
        d.results_folder = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        # 以下容器可能由 notebook 交互流程预先写入，因此只在不存在时初始化。
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.labels_per_cluster = {} if not hasattr(d, "labels_per_cluster") else d.labels_per_cluster
        d.detected_cells = {} if not hasattr(d, "detected_cells") else d.detected_cells
        return d

    def run_strategies(self, d):
        """
        运行全部基础策略或通过历史数据筛选后的策略。

        输入：
            d：Dataset 对象。

        输出：
            无直接返回值，策略画像会写入 d.strategy_profiles。
        """
        sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
        if not self.STRATEGY_FILTERING:
            if os.path.exists(sp_folder_path):
                sys.stderr.write("I just load strategies' results as they have already been run on the dataset!\n")
                # 结果目录存在时优先复用缓存，避免重复运行代价较高的基础检测器。
                strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                          for strategy_file in os.listdir(sp_folder_path)]
            else:
                if self.SAVE_RESULTS:
                    os.mkdir(sp_folder_path)
                algorithm_and_configurations = []
                for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
                    if algorithm_name == "OD":
                        # OD 为 dBoost 生成多组统计检测参数。
                        configuration_list = [
                            list(a) for a in
                            list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                                   ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                            list(itertools.product(["gaussian"],
                                                   ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "PVD":
                        # PVD 为每个列和列中出现过的字符生成一个检测配置。
                        configuration_list = []
                        for attribute in d.dataframe.columns:
                            column_data = "".join(d.dataframe[attribute].tolist())
                            characters_dictionary = {ch: 1 for ch in column_data}
                            for ch in characters_dictionary:
                                configuration_list.append([attribute, ch])
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "RVD":
                        # RVD 枚举不同列之间的映射关系，寻找函数依赖冲突。
                        al = d.dataframe.columns.tolist()
                        configuration_list = [[a, b] for (a, b) in itertools.product(al, al) if a != b]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                    elif algorithm_name == "KBVD":
                        # KBVD 以知识库文件为粒度生成检测配置。
                        configuration_list = [
                            os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base", pat)
                            for pat in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base"))]
                        algorithm_and_configurations.extend(
                            [[d, algorithm_name, configuration] for configuration in configuration_list])
                random.shuffle(algorithm_and_configurations)
                # 多进程并发执行基础检测策略，降低完整策略组合的总耗时。
                pool = multiprocessing.Pool()
                strategy_profiles_list = pool.map(self._strategy_runner_process, algorithm_and_configurations)
                # 如需显式等待进程池回收，可在调试时打开下面两行。
                # pool.close()
                # pool.join()
        else:
            # 策略过滤模式会先构建当前数据和历史数据的画像，再选择更有希望的策略集合。
            for dd in self.HISTORICAL_DATASETS + [d.dictionary]:
                raha.utilities.dataset_profiler(dd)
                raha.utilities.evaluation_profiler(dd)
            strategy_profiles_list = raha.utilities.get_selected_strategies_via_historical_data(d.dictionary, self.HISTORICAL_DATASETS)
        d.strategy_profiles = strategy_profiles_list
        if self.VERBOSE:
            print("{} strategy profiles are collected.".format(len(d.strategy_profiles)))

    def generate_features(self, d):
        """
        根据基础策略输出为每个数据单元格生成特征向量。

        输入：
            d：包含 strategy_profiles 的 Dataset 对象。

        输出：
            无直接返回值，按列生成的特征矩阵会写入 d.column_features。
        """
        columns_features_list = []
        for j in range(d.dataframe.shape[1]):
            feature_vectors = numpy.zeros((d.dataframe.shape[0], len(d.strategy_profiles)))
            for strategy_index, strategy_profile in enumerate(d.strategy_profiles):
                strategy_name = json.loads(strategy_profile["name"])[0]
                if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                    for cell in strategy_profile["output"]:
                        if cell[1] == j:
                            # 某个策略命中当前单元格时，将该策略对应的二值特征置为 1。
                            feature_vectors[cell[0], strategy_index] = 1.0
            if "TFIDF" in self.ERROR_DETECTION_ALGORITHMS:
                # TFIDF 将文本内容本身转为补充特征，适合字符串列较多的数据集。
                vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
                corpus = d.dataframe.iloc[:, j]
                try:
                    tfidf_features = vectorizer.fit_transform(corpus)
                    feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
                except:
                    # 空列或无法向量化的列不使用 TFIDF 特征，保留已有策略特征继续处理。
                    pass
            # 移除所有行完全相同的特征列，避免后续聚类和分类被无信息特征干扰。
            non_identical_columns = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
            feature_vectors = feature_vectors[:, non_identical_columns]
            if self.VERBOSE:
                print("{} Features are generated for column {}.".format(feature_vectors.shape[1], j))
            columns_features_list.append(feature_vectors)
        d.column_features = columns_features_list

    def build_clusters(self, d):
        """
        基于每列特征向量构建层次聚类结果。

        输入：
            d：包含 column_features 的 Dataset 对象。

        输出：
            无直接返回值，聚类到单元格、单元格到聚类的双向索引会写入 Dataset。
        """
        clustering_results = []
        for j in range(d.dataframe.shape[1]):
            feature_vectors = d.column_features[j]
            clusters_k_c_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
            cells_clusters_k_ce = {k: {} for k in range(2, self.LABELING_BUDGET + 2)}
            try:
                # 层次聚类按列构建，cosine 距离用于衡量策略命中特征的相似度。
                clustering_model = scipy.cluster.hierarchy.linkage(feature_vectors, method="average", metric="cosine")
                for k in clusters_k_c_ce:
                    model_labels = [l - 1 for l in
                                    scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                    for index, c in enumerate(model_labels):
                        if c not in clusters_k_c_ce[k]:
                            clusters_k_c_ce[k][c] = {}
                        cell = (index, j)
                        clusters_k_c_ce[k][c][cell] = 1
                        cells_clusters_k_ce[k][cell] = c
            except:
                # 特征为空或距离不可计算时，该列保留空聚类结构，后续流程会跳过。
                pass
            if self.VERBOSE:
                print("A hierarchical clustering model is built for column {}.".format(j))
            clustering_results.append([clusters_k_c_ce, cells_clusters_k_ce])
        # 按不同标注轮次 k 保存聚类结构，便于抽样阶段动态调整簇数量。
        d.clusters_k_j_c_ce = {k: {j: clustering_results[j][0][k] for j in range(d.dataframe.shape[1])} for k in
                               range(2, self.LABELING_BUDGET + 2)}
        d.cells_clusters_k_j_ce = {k: {j: clustering_results[j][1][k] for j in range(d.dataframe.shape[1])} for k in
                                   range(2, self.LABELING_BUDGET + 2)}

    def sample_tuple(self, d):
        """
        从数据集中抽样一条下一轮需要用户标注的元组。

        输入：
            d：包含聚类结果和历史标注状态的 Dataset 对象。

        输出：
            无直接返回值，抽样到的行号会写入 d.sampled_tuple。
        """
        # --------------------计算每个聚类中已有标注数量--------------------
        k = len(d.labeled_tuples) + 2
        for j in range(d.dataframe.shape[1]):
            for c in d.clusters_k_j_c_ce[k][j]:
                d.labels_per_cluster[(j, c)] = {cell: d.labeled_cells[cell][0] for cell in d.clusters_k_j_c_ce[k][j][c] if
                                                cell[0] in d.labeled_tuples}
        # --------------------抽样下一条元组--------------------
        if self.CLUSTERING_BASED_SAMPLING:
            tuple_score = numpy.zeros(d.dataframe.shape[0])
            for i in range(d.dataframe.shape[0]):
                if i not in d.labeled_tuples:
                    score = 0.0
                    for j in range(d.dataframe.shape[1]):
                        if d.clusters_k_j_c_ce[k][j]:
                            cell = (i, j)
                            c = d.cells_clusters_k_j_ce[k][j][cell]
                            # 标注较少的簇会得到更高分数，从而优先探索信息不足的区域。
                            score += math.exp(-len(d.labels_per_cluster[(j, c)]))
                    tuple_score[i] = math.exp(score)
        else:
            # 非聚类模式不区分样本价值，所有未标注元组使用相同权重。
            tuple_score = numpy.ones(d.dataframe.shape[0])
        sum_tuple_score = sum(tuple_score)
        p_tuple_score = tuple_score / sum_tuple_score
        d.sampled_tuple = numpy.random.choice(numpy.arange(d.dataframe.shape[0]), 1, p=p_tuple_score)[0]
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

    def label_with_ground_truth(self, d):
        """
        使用真实干净数据自动标注当前抽样元组。

        输入：
            d：包含 sampled_tuple 和 clean_dataframe 的 Dataset 对象。

        输出：
            无直接返回值，标注结果会写入 d.labeled_tuples 和 d.labeled_cells。

        注意：
            该方法用于离线评测或已有 clean_path 的场景；真实交互场景应由用户人工标注。
        """
        k = len(d.labeled_tuples) + 2
        d.labeled_tuples[d.sampled_tuple] = 1
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            user_label = int(cell in actual_errors_dictionary)
            if random.random() > self.USER_LABELING_ACCURACY:
                # 通过随机翻转模拟用户误标，便于评估标注噪声对结果的影响。
                user_label = 1 - user_label
            d.labeled_cells[cell] = [user_label, d.clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(d.sampled_tuple))

    def propagate_labels(self, d):
        """
        将用户标注从已标注单元格传播到同簇的其他单元格。

        输入：
            d：包含聚类结果和已标注单元格的 Dataset 对象。

        输出：
            无直接返回值，扩展后的标签会写入 d.extended_labeled_cells。
        """
        # 先保留用户直接标注结果，后续传播只在此基础上增加标签。
        d.extended_labeled_cells = {cell: d.labeled_cells[cell][0] for cell in d.labeled_cells}
        k = len(d.labeled_tuples) + 2 - 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            if cell in d.cells_clusters_k_j_ce[k][j]:
                c = d.cells_clusters_k_j_ce[k][j][cell]
                d.labels_per_cluster[(j, c)][cell] = d.labeled_cells[cell][0]
        if self.CLUSTERING_BASED_SAMPLING:
            for j in d.clusters_k_j_c_ce[k]:
                for c in d.clusters_k_j_c_ce[k][j]:
                    if len(d.labels_per_cluster[(j, c)]) > 0:
                        if self.LABEL_PROPAGATION_METHOD == "homogeneity":
                            # homogeneity 只在簇内已标注样本完全一致时传播，精度更保守。
                            cluster_label = list(d.labels_per_cluster[(j, c)].values())[0]
                            if sum(d.labels_per_cluster[(j, c)].values()) in [0, len(d.labels_per_cluster[(j, c)])]:
                                for cell in d.clusters_k_j_c_ce[k][j][c]:
                                    d.extended_labeled_cells[cell] = cluster_label
                        elif self.LABEL_PROPAGATION_METHOD == "majority":
                            # majority 使用多数投票传播，覆盖更多单元格但更依赖聚类质量。
                            cluster_label = round(
                                sum(d.labels_per_cluster[(j, c)].values()) / len(d.labels_per_cluster[(j, c)]))
                            for cell in d.clusters_k_j_c_ce[k][j][c]:
                                d.extended_labeled_cells[cell] = cluster_label
        if self.VERBOSE:
            print("The number of labeled data cells increased from {} to {}.".format(len(d.labeled_cells), len(d.extended_labeled_cells)))

    def predict_labels(self, d):
        """
        训练列级分类器并预测所有数据单元格是否为错误。

        输入：
            d：包含特征矩阵和扩展标签的 Dataset 对象。

        输出：
            无直接返回值，预测出的错误单元格会合并到 d.detected_cells。
        """
        detected_cells_dictionary = {}
        for j in range(d.dataframe.shape[1]):
            feature_vectors = d.column_features[j]
            x_train = [feature_vectors[i, :] for i in range(d.dataframe.shape[0]) if (i, j) in d.extended_labeled_cells]
            y_train = [d.extended_labeled_cells[(i, j)] for i in range(d.dataframe.shape[0]) if
                       (i, j) in d.extended_labeled_cells]
            x_test = feature_vectors
            if sum(y_train) == len(y_train):
                # 已知训练样本全为错误时，当前列全部预测为错误。
                predicted_labels = numpy.ones(d.dataframe.shape[0])
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                # 没有正样本或没有有效特征时，当前列全部预测为正确。
                predicted_labels = numpy.zeros(d.dataframe.shape[0])
            else:
                # 根据配置选择分类器，训练样本来自用户直接标注和聚类传播标签。
                if self.CLASSIFICATION_MODEL == "ABC":
                    classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                if self.CLASSIFICATION_MODEL == "DTC":
                    classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                if self.CLASSIFICATION_MODEL == "GBC":
                    classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                if self.CLASSIFICATION_MODEL == "GNB":
                    classification_model = sklearn.naive_bayes.GaussianNB()
                if self.CLASSIFICATION_MODEL == "KNC":
                    classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                if self.CLASSIFICATION_MODEL == "SGDC":
                    classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                if self.CLASSIFICATION_MODEL == "SVC":
                    classification_model = sklearn.svm.SVC(kernel="sigmoid")
                classification_model.fit(x_train, y_train)
                predicted_labels = classification_model.predict(x_test)
            for i, pl in enumerate(predicted_labels):
                if (i in d.labeled_tuples and d.extended_labeled_cells[(i, j)]) or (i not in d.labeled_tuples and pl):
                    # 直接标注为错误或模型预测为错误的单元格都会进入最终检测结果。
                    detected_cells_dictionary[(i, j)] = "JUST A DUMMY VALUE"
            if self.VERBOSE:
                print("A classifier is trained and applied on column {}.".format(j))
        d.detected_cells.update(detected_cells_dictionary)

    def store_results(self, d):
        """
        将检测流程的 Dataset 对象保存到结果目录。

        输入：
            d：已完成检测流程的 Dataset 对象。

        输出：
            无直接返回值，会写入 error-detection/detection.dataset 文件。
        """
        ed_folder_path = os.path.join(d.results_folder, "error-detection")
        if not os.path.exists(ed_folder_path):
            os.mkdir(ed_folder_path)
        # 保存完整 Dataset 便于后续分析策略输出、特征、聚类和最终检测结果。
        pickle.dump(d, open(os.path.join(ed_folder_path, "detection.dataset"), "wb"))
        if self.VERBOSE:
            print("The results are stored in {}.".format(os.path.join(ed_folder_path, "detection.dataset")))

    def run(self, dd):
        """
        对输入数据集执行完整 Raha 错误检测流程。

        输入：
            dd：数据集字典。

        输出：
            detected_cells：错误单元格字典，键为 (row_index, column_index)。

        示例：
            app = Detection()
            app.LABELING_BUDGET = 10
            app.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD"]
            detected_cells = app.run(dataset_dictionary)
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initializing the Dataset Object--------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(dd)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------------Running Error Detection Strategies-------------------\n"
                  "------------------------------------------------------------------------")
        self.run_strategies(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-----------------------Generating Feature Vectors-----------------------\n"
                  "------------------------------------------------------------------------")
        self.generate_features(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------Building the Hierarchical Clustering Model---------------\n"
                  "------------------------------------------------------------------------")
        self.build_clusters(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "-------------Iterative Clustering-Based Sampling and Labeling-----------\n"
                  "------------------------------------------------------------------------")
        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(d)
            if d.has_ground_truth:
                # 有真实干净数据时直接自动标注，适合 benchmark 和回归测试。
                self.label_with_ground_truth(d)
            # 否则：
            #   没有真实标签时，需要按 Jupyter notebook 中的交互流程由用户标注当前元组。
            if self.VERBOSE:
                print("------------------------------------------------------------------------")
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Propagating User Labels Through the Clusters--------------\n"
                  "------------------------------------------------------------------------")
        self.propagate_labels(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------Training and Testing Classification Models---------------\n"
                  "------------------------------------------------------------------------")
        self.predict_labels(d)
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            self.store_results(d)
        return d.detected_cells
########################################


########################################
if __name__ == "__main__":
    # 命令行直接运行本文件时，使用内置 flights 数据集演示完整检测与评估流程。
    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = Detection()
    detection_dictionary = app.run(dataset_dictionary)
    data = raha.dataset.Dataset(dataset_dictionary)
    p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]
    print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
    # --------------------历史数据策略过滤示例--------------------
    # 以下代码展示如何开启历史数据策略过滤。
    # app.STRATEGY_FILTERING = True
    # app.HISTORICAL_DATASETS = [
    #     {
    #         "name": "hospital",
    #         "path": "/path/to/hospital/dirty.csv",
    #         "clean_path": "/path/to/hospital/clean.csv"
    #     },
    #     {
    #         "name": "beers",
    #         "path": "/path/to/beers/dirty.csv",
    #         "clean_path": "/path/to/beers/clean.csv"
    #     }
    # ]
    # detection_dictionary = app.run(dataset_dictionary)
#######################################
