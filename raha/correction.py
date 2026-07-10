########################################
# Baran: The Error Correction System
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
# 本模块实现 Baran 的错误修复主流程。
#
# 基础使用示例：
#     import os
#     import raha
#     from raha.correction import Correction
#
#     dataset_name = "flights"
#     dataset_dictionary = {
#         "name": dataset_name,
#         "path": os.path.abspath("datasets/flights/dirty.csv"),
#         "clean_path": os.path.abspath("datasets/flights/clean.csv"),
#     }
#
#     data = raha.dataset.Dataset(dataset_dictionary)
#     data.detected_cells = dict(data.get_actual_errors_dictionary())
#
#     app = Correction()
#     app.LABELING_BUDGET = 20
#     app.CLASSIFICATION_MODEL = "ABC"
#     app.VERBOSE = True
#     corrected_cells = app.run(data)
#     print(corrected_cells)
#
# 与 Raha 检测结果串联示例：
#     detector = raha.detection.Detection()
#     data = raha.dataset.Dataset(dataset_dictionary)
#     data.detected_cells = detector.run(dataset_dictionary)
#     corrected_cells = Correction().run(data)
#
# 离线预训练示例：
#     app = Correction()
#     app.MIN_CORRECTION_OCCURRENCE = 2
#     app.MAX_VALUE_LENGTH = 50
#     app.extract_revisions(wikipedia_dumps_folder="/path/to/wikipedia-data")
#     app.pretrain_value_based_models(revision_data_folder="/path/to/wikipedia-data/revision-data")
#
# 可行配置重点说明：
#     PRETRAINED_VALUE_BASED_MODELS_PATH：
#         预训练 value-based 模型路径；为空时只使用当前数据集在线学习。
#     VALUE_ENCODINGS：
#         可选 "identity"、"unicode"；默认同时使用原值字符和 Unicode 类别抽象。
#     CLASSIFICATION_MODEL：
#         可选 "ABC"、"DTC"、"GBC"、"GNB"、"KNC"、"SGDC"、"SVC"。
#         小样本默认推荐 "ABC"；若需要 decision_function 置信度，可优先考虑 "SGDC" 或 "SVC"。
#     USE_PREDICTION_CONFIDENCE：
#         为 True 且分类器支持 decision_function 时，选择置信度最高的正类候选修复。
#     LABELING_BUDGET：
#         用户标注轮数，建议为正整数且不超过数据行数。
#     MIN_CORRECTION_CANDIDATE_PROBABILITY：
#         候选修复最小概率阈值，范围建议为 0.0 到 1.0；越高候选越少。
#     MIN_CORRECTION_OCCURRENCE：
#         离线预训练时的最小出现次数，值越大模型越保守。
#     MAX_VALUE_LENGTH：
#         离线预训练时参与建模的值长度上限，避免长文本噪声和内存膨胀。
#     REVISION_WINDOW_SIZE：
#         抽取维基修订时保留的上下文窗口大小。
#     NUM_WORKERS、CHUNK_SIZE：
#         并行预测进程数和每批单元格数量；资源紧张时可调小。
#     IGNORE_SIGN、ONLINE_PHASE：
#         内部哨兵值和流程状态，通常不建议业务侧修改。
########################################


import bz2
import difflib
import functools
import io
import itertools
import json
import math
########################################
import os
import pickle
import sys
import unicodedata
from multiprocessing import Pool

import bs4
import mwparserfromhell
import numpy
import py7zr
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree

import raha

########################################

def worker_init_prediction(dataset, cls_model):
    """
    初始化预测子进程的共享数据集和分类器。

    multiprocessing 需要通过模块级全局变量把大对象传给 worker，避免每个任务重复传输。
    """
    global d
    d = dataset 
    global classification_model
    classification_model = cls_model

def worker_init_feat_generation(dataset):
    """
    初始化特征生成子进程的共享数据集。

    输入：
        dataset：已经初始化模型和标注状态的 Dataset 对象。
    """
    global d
    d = dataset 

########################################
class Correction:
    """
    Baran 错误修复主类。

    该类以错误检测结果为输入，生成候选修复值，并通过用户标注样本训练分类器，
    最终输出数据单元格到修复值的映射。

    输入：
        d：Dataset 对象，必须包含 detected_cells。

    输出：
        run 方法返回 corrected_cells 字典，键为 (row_index, column_index)，值为修复后的字符串。
    """

    def __init__(self):
        """
        初始化 Baran 错误修复流程的默认配置。

        常用可调项是 LABELING_BUDGET、CLASSIFICATION_MODEL、PRETRAINED_VALUE_BASED_MODELS_PATH、
        MIN_CORRECTION_CANDIDATE_PROBABILITY、NUM_WORKERS 和 CHUNK_SIZE。
        """
        # 预训练 value-based 模型文件路径；为空时仅使用当前数据集在线学习。
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        # 值编码方式：identity 保留字符本身，unicode 使用字符类别抽象；默认两者互补使用。
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        # 是否启用基于同一行邻近上下文的候选修复模型；关闭后仅使用值模型和列值域模型。
        self.USE_VICINITY_BASED_MODEL = True
        # 候选修复二分类模型；可选 ABC、DTC、GBC、GNB、KNC、SGDC、SVC。
        self.CLASSIFICATION_MODEL = "ABC"   # ["ABC", "DTC", "GBC", "GNB", "KNC" ,"SGDC", "SVC"]
        # 分类器支持 decision_function 时，是否使用置信度选择最可信的候选修复。
        self.USE_PREDICTION_CONFIDENCE = True
        # 内部哨兵值，用于表示上下文中应忽略的错误值或当前列值。
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        # 是否输出详细过程信息。
        self.VERBOSE = False
        # 是否保存完整修复结果对象到结果目录。
        self.SAVE_RESULTS = True
        # 内部流程状态：run 阶段会置为 True，预训练阶段通常保持 False。
        self.ONLINE_PHASE = False
        # 用户标注轮数，建议为正整数且不超过数据集行数。
        self.LABELING_BUDGET = 20
        # 候选修复最小概率阈值，0.0 表示不过滤低概率候选。
        self.MIN_CORRECTION_CANDIDATE_PROBABILITY = 0.0
        # 离线预训练时保留修复模式的最小出现次数。
        self.MIN_CORRECTION_OCCURRENCE = 2
        # 离线预训练时参与建模的值长度上限，避免长文本带来噪声和内存压力。
        self.MAX_VALUE_LENGTH = 50
        # 抽取维基修订差异时保留的左右上下文片段数量。
        self.REVISION_WINDOW_SIZE = 5
        # 并行特征生成和预测使用的进程数；资源有限时可调小。
        self.NUM_WORKERS = os.cpu_count()
        # 每个子进程批次处理的错误单元格数量。
        self.CHUNK_SIZE = 100

    @staticmethod
    def _wikitext_segmenter(wikitext):
        """
        递归解析 Wikipedia wikitext，并切分出可比较的文本片段。

        输入：
            wikitext：单个页面修订版本的 wikitext 字符串。

        输出：
            segments_list：按结构展开后的文本片段列表。
        """
        def recursive_segmenter(node):
            """
            递归遍历 mwparserfromhell 节点，把可见文本追加到 segments_list。
            """
            if isinstance(node, str):
                segments_list.append(node)
            elif isinstance(node, mwparserfromhell.nodes.text.Text):
                segments_list.append(node.value)
            elif not node:
                pass
            elif isinstance(node, mwparserfromhell.wikicode.Wikicode):
                for n in node.nodes:
                    if isinstance(n, str):
                        recursive_segmenter(n)
                    elif isinstance(n, mwparserfromhell.nodes.text.Text):
                        recursive_segmenter(n.value)
                    elif isinstance(n, mwparserfromhell.nodes.heading.Heading):
                        recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.tag.Tag):
                        recursive_segmenter(n.contents)
                    elif isinstance(n, mwparserfromhell.nodes.wikilink.Wikilink):
                        if n.text:
                            recursive_segmenter(n.text)
                        else:
                            recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.external_link.ExternalLink):
                        # 外部链接只取标题文本，不把 URL 本身作为修复训练片段。
                        recursive_segmenter(n.title)
                    elif isinstance(n, mwparserfromhell.nodes.template.Template):
                        recursive_segmenter(n.name)
                        for p in n.params:
                            # 模板参数名通常重复且噪声较高，这里只解析参数值。
                            recursive_segmenter(p.value)
                    elif isinstance(n, mwparserfromhell.nodes.html_entity.HTMLEntity):
                        segments_list.append(n.normalize())
                    elif not n or isinstance(n, mwparserfromhell.nodes.comment.Comment) or \
                            isinstance(n, mwparserfromhell.nodes.argument.Argument):
                        pass
                    else:
                        sys.stderr.write("Inner layer unknown node found: {}, {}\n".format(type(n), n))
            else:
                sys.stderr.write("Outer layer unknown node found: {}, {}\n".format(type(node), node))

        try:
            parsed_wikitext = mwparserfromhell.parse(wikitext)
        except:
            # 单个修订解析失败时返回空片段，避免中断整个 dump 抽取流程。
            parsed_wikitext = ""
        segments_list = []
        recursive_segmenter(parsed_wikitext)
        return segments_list

    def extract_revisions(self, wikipedia_dumps_folder):
        """
        从 Wikipedia 页面修订历史 dump 中抽取 value-based 修复训练数据。

        输入：
            wikipedia_dumps_folder：包含 .7z 压缩 dump 的目录。

        输出：
            无直接返回值，会在 revision-data 子目录写入按页面保存的 JSON 修订差异。

        示例：
            app = Correction()
            app.REVISION_WINDOW_SIZE = 5
            app.extract_revisions("/path/to/wikipedia-data")
        """
        rd_folder_path = os.path.join(wikipedia_dumps_folder, "revision-data")
        if not os.path.exists(rd_folder_path):
            os.mkdir(rd_folder_path)
        # 每个 .7z 文件通常包含一个解压后的 XML 修订历史文件。
        compressed_dumps_list = [df for df in os.listdir(wikipedia_dumps_folder) if df.endswith(".7z")]
        page_counter = 0
        for file_name in compressed_dumps_list:
            compressed_dump_file_path = os.path.join(wikipedia_dumps_folder, file_name)
            dump_file_name, _ = os.path.splitext(os.path.basename(compressed_dump_file_path))
            rdd_folder_path = os.path.join(rd_folder_path, dump_file_name)
            if not os.path.exists(rdd_folder_path):
                os.mkdir(rdd_folder_path)
            else:
                # 已抽取过的 dump 目录直接跳过，便于断点续跑。
                continue
            # 外部文件系统读写较重，处理完成后会删除解压出的临时 XML 文件。
            archive = py7zr.SevenZipFile(compressed_dump_file_path, mode="r")
            archive.extractall(path=wikipedia_dumps_folder)
            archive.close()
            decompressed_dump_file_path = os.path.join(wikipedia_dumps_folder, dump_file_name)
            decompressed_dump_file = io.open(decompressed_dump_file_path, "r", encoding="utf-8")
            page_text = ""
            for i, line in enumerate(decompressed_dump_file):
                line = line.strip()
                if line == "<page>":
                    page_text = ""
                page_text += "\n" + line
                if line == "</page>":
                    revisions_list = []
                    page_tree = bs4.BeautifulSoup(page_text, "html.parser")
                    previous_text = ""
                    for revision_tag in page_tree.find_all("revision"):
                        revision_text = revision_tag.find("text").text
                        if previous_text:
                            # 相邻修订版本做序列差异，非 equal 片段作为潜在修复样本。
                            a = [t for t in self._wikitext_segmenter(previous_text) if t]
                            b = [t for t in self._wikitext_segmenter(revision_text) if t]
                            s = difflib.SequenceMatcher(None, a, b)
                            for tag, i1, i2, j1, j2 in s.get_opcodes():
                                if tag == "equal":
                                    continue
                                revisions_list.append({
                                    "old_value": a[i1:i2],
                                    "new_value": b[j1:j2],
                                    "left_context": a[i1 - self.REVISION_WINDOW_SIZE:i1],
                                    "right_context": a[i2:i2 + self.REVISION_WINDOW_SIZE]
                                })
                        previous_text = revision_text
                    if revisions_list:
                        page_counter += 1
                        if self.VERBOSE and page_counter % 100 == 0:
                            for entry in revisions_list:
                                print("----------Page Counter:---------\n", page_counter,
                                      "\n----------Old Value:---------\n", entry["old_value"],
                                      "\n----------New Value:---------\n", entry["new_value"],
                                      "\n----------Left Context:---------\n", entry["left_context"],
                                      "\n----------Right Context:---------\n", entry["right_context"],
                                      "\n==============================")
                        json.dump(revisions_list, open(os.path.join(rdd_folder_path, page_tree.id.text + ".json"), "w"))
            decompressed_dump_file.close()
            os.remove(decompressed_dump_file_path)
            if self.VERBOSE:
                print("{} ({} / {}) is processed.".format(file_name, len(os.listdir(rd_folder_path)), len(compressed_dumps_list)))

    @staticmethod
    def _value_encoder(value, encoding):
        """
        按指定编码方式表示单元格值。

        可行配置：
            identity：保留原字符序列，适合学习固定字符串替换。
            unicode：使用 Unicode 字符类别，适合学习格式类修复。
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        向字典实现的统计模型增量累加一次 key 到 value 的观察。

        输入：
            model：嵌套字典模型。
            key：外层键。
            value：内层键。
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    def _value_based_models_updater(self, models, ud):
        """
        使用一条修复样本更新 value-based 修复模型。

        输入：
            models：remover、adder、replacer、swapper 四类值模型。
            ud：包含 old_value、new_value 的更新字典。
        """
        # TODO：可继续扩展子串交换类修复模式。
        if self.ONLINE_PHASE or (ud["new_value"] and len(ud["new_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] and len(ud["old_value"]) <= self.MAX_VALUE_LENGTH and
                                 ud["old_value"] != ud["new_value"] and ud["old_value"].lower() != "n/a" and
                                 not ud["old_value"][0].isdigit()):
            # 离线预训练阶段过滤过长值、空值、无变化值、常见占位值和数字开头值，以降低噪声。
            remover_transformation = {}
            adder_transformation = {}
            replacer_transformation = {}
            s = difflib.SequenceMatcher(None, ud["old_value"], ud["new_value"])
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                index_range = json.dumps([i1, i2])
                if tag == "delete":
                    # 删除型修复：把旧值某个区间删除。
                    remover_transformation[index_range] = ""
                if tag == "insert":
                    # 插入型修复：在旧值某个位置插入新片段。
                    adder_transformation[index_range] = ud["new_value"][j1:j2]
                if tag == "replace":
                    # 替换型修复：删除旧区间并插入新片段。
                    replacer_transformation[index_range] = ud["new_value"][j1:j2]
            for encoding in self.VALUE_ENCODINGS:
                encoded_old_value = self._value_encoder(ud["old_value"], encoding)
                if remover_transformation:
                    self._to_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation))
                if adder_transformation:
                    self._to_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation))
                if replacer_transformation:
                    self._to_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation))
                self._to_model_adder(models[3], encoded_old_value, ud["new_value"])

    def pretrain_value_based_models(self, revision_data_folder):
        """
        基于已抽取的修订数据预训练 value-based 修复模型。

        输入：
            revision_data_folder：extract_revisions 生成的 revision-data 目录。

        输出：
            无直接返回值，会写入 pretrained_value_based_models.dictionary。

        可行配置：
            MIN_CORRECTION_OCCURRENCE 控制模型剪枝强度。
            PRETRAINED_VALUE_BASED_MODELS_PATH 可指定模型输出路径。
            VALUE_ENCODINGS 控制预训练时使用哪些值编码。
        """
        def _models_pruner():
            """
            删除出现次数不足的修复模式，降低预训练模型体积和噪声。
            """
            for mi, model in enumerate(models):
                for k in list(model.keys()):
                    for v in list(model[k].keys()):
                        if model[k][v] < self.MIN_CORRECTION_OCCURRENCE:
                            models[mi][k].pop(v)
                    if not models[mi][k]:
                        models[mi].pop(k)

        models = [{}, {}, {}, {}]
        rd_folder_path = revision_data_folder
        page_counter = 0
        for folder in os.listdir(rd_folder_path):
            if os.path.isdir(os.path.join(rd_folder_path, folder)):
                for rf in os.listdir(os.path.join(rd_folder_path, folder)):
                    if rf.endswith(".json"):
                        page_counter += 1
                        if page_counter % 100000 == 0:
                            _models_pruner()
                            if self.VERBOSE:
                                print(page_counter, "pages are processed.")
                        try:
                            # 单个页面修订 JSON 损坏时跳过，避免影响长时间批处理任务。
                            revision_list = json.load(io.open(os.path.join(rd_folder_path, folder, rf), encoding="utf-8"))
                        except:
                            continue
                        for rd in revision_list:
                            update_dictionary = {
                                "old_value": raha.dataset.Dataset.value_normalizer("".join(rd["old_value"])),
                                "new_value": raha.dataset.Dataset.value_normalizer("".join(rd["new_value"]))
                            }
                            self._value_based_models_updater(models, update_dictionary)
        _models_pruner()
        pretrained_models_path = os.path.join(revision_data_folder, "pretrained_value_based_models.dictionary")
        if self.PRETRAINED_VALUE_BASED_MODELS_PATH:
            pretrained_models_path = self.PRETRAINED_VALUE_BASED_MODELS_PATH
        pickle.dump(models, bz2.BZ2File(pretrained_models_path, "wb"))
        if self.VERBOSE:
            print("The pretrained value-based models are stored in {}.".format(pretrained_models_path))

    def _vicinity_based_models_updater(self, models, ud):
        """
        使用上下文值更新 vicinity-based 修复模型。

        输入：
            models：上下文列到目标列的统计模型。
            ud：包含 column、new_value、vicinity 的更新字典。
        """
        for j, cv in enumerate(ud["vicinity"]):
            if cv != self.IGNORE_SIGN:
                self._to_model_adder(models[j][ud["column"]], cv, ud["new_value"])

    def _domain_based_model_updater(self, model, ud):
        """
        使用目标列的新值更新 domain-based 修复模型。

        domain 模型学习每一列中常见的合法取值分布。
        """
        self._to_model_adder(model, ud["column"], ud["new_value"])

    def _value_based_corrector(self, models, ed):
        """
        基于 value-based 模型为一个错误单元格生成候选修复。

        输入：
            models：四类值模型。
            ed：包含 column、old_value、vicinity 的错误字典。

        输出：
            results_list：每个子模型输出一个候选修复到概率的字典。
        """
        results_list = []
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in self.VALUE_ENCODINGS:
                results_dictionary = {}
                encoded_value_string = self._value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            # 将学习到的字符级变换应用到当前旧值，得到一个候选新值。
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            pr = model[encoded_value_string][transformation_string] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                    if model_name == "swapper":
                        # swapper 直接学习 old_value 到 new_value 的整体替换。
                        for new_value in model[encoded_value_string]:
                            pr = model[encoded_value_string][new_value] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                results_list.append(results_dictionary)
        return results_list

    def _vicinity_based_corrector(self, models, ed):
        """
        基于同一行其他列的上下文值生成候选修复。

        输出：
            results_list：每个上下文列对应一个候选修复字典。
        """
        results_list = []
        for j, cv in enumerate(ed["vicinity"]):
            results_dictionary = {}
            if j != ed["column"] and cv in models[j][ed["column"]]:
                sum_scores = sum(models[j][ed["column"]][cv].values())
                for new_value in models[j][ed["column"]][cv]:
                    pr = models[j][ed["column"]][cv][new_value] / sum_scores
                    if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                        results_dictionary[new_value] = pr
            results_list.append(results_dictionary)
        return results_list

    def _domain_based_corrector(self, model, ed):
        """
        基于目标列值域分布生成候选修复。

        domain 模型不依赖旧值或上下文，仅从同列常见值中给出候选。
        """
        results_dictionary = {}
        sum_scores = sum(model[ed["column"]].values())
        for new_value in model[ed["column"]]:
            pr = model[ed["column"]][new_value] / sum_scores
            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                results_dictionary[new_value] = pr
        return [results_dictionary]

    def initialize_dataset(self, d):
        """
        初始化修复流程需要的数据集状态。

        输入：
            d：Dataset 对象，必须已经设置 detected_cells。

        输出：
            d：补充结果目录、错误列索引和标注容器后的 Dataset 对象。
        """
        # run 阶段进入在线学习模式，此时真实标注样本不再使用离线预训练过滤规则。
        self.ONLINE_PHASE = True
        d.results_folder = os.path.join(os.path.dirname(d.path), "raha-baran-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        d.column_errors = {}
        for cell in d.detected_cells:
            # 按列组织错误单元格，后续每列训练一个候选修复分类器。
            self._to_model_adder(d.column_errors, cell[1], cell)
        # notebook 交互流程可能已写入这些容器，因此只在不存在时初始化。
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells
        return d

    def initialize_models(self, d):
        """
        初始化三类错误修复模型。

        模型类型：
            value_models：从旧值到新值或字符变换的模型。
            vicinity_models：从同一行上下文值到目标列新值的模型。
            domain_models：从列到常见合法值的模型。
        """
        d.value_models = [{}, {}, {}, {}]
        if os.path.exists(self.PRETRAINED_VALUE_BASED_MODELS_PATH):
            # 外部预训练模型是可选配置，存在时会增强 value-based 候选生成能力。
            d.value_models = pickle.load(bz2.BZ2File(self.PRETRAINED_VALUE_BASED_MODELS_PATH, "rb"))
            if self.VERBOSE:
                print("The pretrained value-based models are loaded.")
        d.vicinity_models = {}
        if self.USE_VICINITY_BASED_MODEL:
            # 邻近上下文模型按“上下文列 -> 目标列”维护统计映射，关闭开关时不分配该结构。
            d.vicinity_models = {j: {jj: {} for jj in range(d.dataframe.shape[1])} for j in range(d.dataframe.shape[1])}
        d.domain_models = {}
        for row in d.dataframe.itertuples():
            i, row = row[0], row[1:]
            if self.USE_VICINITY_BASED_MODEL:
                # 已检测为错误的上下文值不能作为邻近模型信号，避免把脏值学习进去。
                vicinity_list = [cv if (i, cj) not in d.detected_cells else self.IGNORE_SIGN for cj, cv in enumerate(row)]
            for j, value in enumerate(row):
                if (i, j) not in d.detected_cells:
                    # 只用未检测为错误的单元格初始化上下文和列值域模型，降低脏值污染。
                    update_dictionary = {
                        "column": j,
                        "new_value": value,
                    }
                    if self.USE_VICINITY_BASED_MODEL:
                        temp_vicinity_list = list(vicinity_list)
                        temp_vicinity_list[j] = self.IGNORE_SIGN
                        update_dictionary["vicinity"] = temp_vicinity_list
                        self._vicinity_based_models_updater(d.vicinity_models, update_dictionary)
                    self._domain_based_model_updater(d.domain_models, update_dictionary)
        if self.VERBOSE:
            print("The error corrector models are initialized.")

    def sample_tuple(self, d):
        """
        从剩余未修复错误中抽样下一条需要用户标注的元组。

        输入：
            d：包含 detected_cells、corrected_cells 和 column_errors 的 Dataset 对象。

        输出：
            无直接返回值，抽样行号会写入 d.sampled_tuple。
        """
        remaining_column_erroneous_cells = {}
        remaining_column_erroneous_values = {}
        for j in d.column_errors:
            for cell in d.column_errors[j]:
                if cell not in d.corrected_cells:
                    self._to_model_adder(remaining_column_erroneous_cells, j, cell)
                    self._to_model_adder(remaining_column_erroneous_values, j, d.dataframe.iloc[cell])
        tuple_score = numpy.ones(d.dataframe.shape[0])
        tuple_score[list(d.labeled_tuples.keys())] = 0.0
        for j in remaining_column_erroneous_cells:
            for cell in remaining_column_erroneous_cells[j]:
                value = d.dataframe.iloc[cell]
                column_score = math.exp(len(remaining_column_erroneous_cells[j]) / len(d.column_errors[j]))
                cell_score = math.exp(remaining_column_erroneous_values[j][value] / len(remaining_column_erroneous_cells[j]))
                # 优先选择错误较多的列和重复出现的错误值，提高一次标注带来的学习收益。
                tuple_score[cell[0]] *= column_score * cell_score
        d.sampled_tuple = numpy.random.choice(numpy.argwhere(tuple_score == numpy.amax(tuple_score)).flatten())
        if self.VERBOSE:
            print("Tuple {} is sampled.".format(d.sampled_tuple))

    def label_with_ground_truth(self, d):
        """
        使用干净数据自动标注当前抽样元组。

        输入：
            d：包含 clean_dataframe 的 Dataset 对象。

        输出：
            无直接返回值，标注结果写入 d.labeled_tuples 和 d.labeled_cells。
        """
        d.labeled_tuples[d.sampled_tuple] = 1
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            error_label = 0
            if d.dataframe.iloc[cell] != d.clean_dataframe.iloc[cell]:
                # 脏值和干净值不同，说明该单元格需要修复。
                error_label = 1
            d.labeled_cells[cell] = [error_label, d.clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(d.sampled_tuple))

    def update_models(self, d):
        """
        用新标注元组更新三类修复模型。

        输入：
            d：包含 sampled_tuple 和 labeled_cells 的 Dataset 对象。

        输出：
            无直接返回值，模型和错误集合会原地更新。
        """
        cleaned_sampled_tuple = [d.labeled_cells[(d.sampled_tuple, j)][1] for j in range(d.dataframe.shape[1])]
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            update_dictionary = {
                "column": cell[1],
                "old_value": d.dataframe.iloc[cell],
                "new_value": cleaned_sampled_tuple[j],
            }
            if d.labeled_cells[cell][0] == 1:
                if cell not in d.detected_cells:
                    # 用户标出了检测阶段遗漏的错误，将其补入待修复集合。
                    d.detected_cells[cell] = self.IGNORE_SIGN
                    self._to_model_adder(d.column_errors, cell[1], cell)
                self._value_based_models_updater(d.value_models, update_dictionary)
                self._domain_based_model_updater(d.domain_models, update_dictionary)
                if self.USE_VICINITY_BASED_MODEL:
                    # 错误单元格使用整行清洗后的上下文，当前列自身仍需忽略。
                    update_dictionary["vicinity"] = [cv if j != cj else self.IGNORE_SIGN
                                                     for cj, cv in enumerate(cleaned_sampled_tuple)]
            elif self.USE_VICINITY_BASED_MODEL:
                # 正确单元格只利用同一元组中被确认修复过的上下文，避免把未确认值作为强信号。
                update_dictionary["vicinity"] = [cv if j != cj and d.labeled_cells[(d.sampled_tuple, cj)][0] == 1
                                                 else self.IGNORE_SIGN for cj, cv in enumerate(cleaned_sampled_tuple)]
            if self.USE_VICINITY_BASED_MODEL:
                self._vicinity_based_models_updater(d.vicinity_models, update_dictionary)
        if self.VERBOSE:
            print("The error corrector models are updated with new labeled tuple {}.".format(d.sampled_tuple))

    def _feature_generator_process(self, cell_list, dataset=None):
        """
        在子进程中为错误单元格生成候选修复特征。

        输入：
            cell_list：错误单元格列表，可能包含 zip_longest 填充的 None。
            dataset：可选 Dataset；为空时使用 worker 初始化的全局 Dataset。

        输出：
            pairs_counter：候选修复对数量。
            pair_features：错误单元格到候选修复特征向量的映射。
            ret_cells：实际处理的单元格列表。
        """
        pairs_counter = 0
        pair_features = {}
        ret_cells = []
        if dataset == None:
            global d
        else:
            d = dataset

        for cell in filter(lambda cell: cell is not None, cell_list):
            error_dictionary = {"column": cell[1], "old_value": d.dataframe.iloc[cell], "vicinity": list(d.dataframe.iloc[cell[0], :])}
            # 三类模型分别生成候选修复，并把每个候选在各模型中的概率拼成特征向量。
            value_corrections = self._value_based_corrector(d.value_models, error_dictionary)
            vicinity_corrections = []
            if self.USE_VICINITY_BASED_MODEL:
                vicinity_corrections = self._vicinity_based_corrector(d.vicinity_models, error_dictionary)
            domain_corrections = self._domain_based_corrector(d.domain_models, error_dictionary)
            models_corrections = value_corrections + vicinity_corrections + domain_corrections
            corrections_features = {}
            for mi, model in enumerate(models_corrections):
                for correction in model:
                    if correction not in corrections_features:
                        corrections_features[correction] = numpy.zeros(len(models_corrections))
                    corrections_features[correction][mi] = model[correction]

            pair_features[cell] = {}
            ret_cells.append(cell)
            for correction in corrections_features:
                pair_features[cell][correction] = corrections_features[correction]
                pairs_counter += 1

        return pairs_counter, pair_features, ret_cells

    def generate_features(self, d, cells):
        """
        为一批错误单元格生成候选修复特征。

        输入：
            d：Dataset 对象。
            cells：待生成特征的错误单元格列表。

        输出：
            生成器，每次产出一个批次的 pair_features 和 cells。

        可行配置：
            NUM_WORKERS 控制并行进程数；CHUNK_SIZE 控制单批任务大小。
        """
        if len(cells) == 0:
            yield {}, []
        else:
            # 特征生成是 CPU 密集型任务，按批次分发到多个进程并行处理。
            pool = Pool(max(self.NUM_WORKERS-1, 1), initargs=(d,), initializer=worker_init_feat_generation)
            pairs_counter = 0
            process_args_generator = itertools.zip_longest(*[iter(cells)] * self.CHUNK_SIZE)

            feature_generation_iterator = pool.imap(self._feature_generator_process, process_args_generator)

            for pairs_counter_out, pair_features_out, cell_list in feature_generation_iterator:
                pairs_counter += pairs_counter_out
                yield pair_features_out, cell_list

            pool.close()
            if self.VERBOSE:
                print("{} pairs of (a data error, a potential correction) are featurized.".format(pairs_counter))


    def _prediction_process(self, cell_list, all_ones, all_zeros, dataset=None, cls_model=None):
        """
        在子进程中预测一批错误单元格的修复值。

        输入：
            cell_list：错误单元格列表。
            all_ones：训练候选是否全为正类。
            all_zeros：训练候选是否全为负类。
            dataset：可选 Dataset。
            cls_model：可选分类器。

        输出：
            correction_dict：单元格到预测修复值的映射。
        """
        if dataset == None:
            global d
        else:
            d = dataset

        if cls_model == None:
            global classification_model
        else:
            classification_model = cls_model
        
        correction_dict = {}

        for cell in filter(lambda cell: cell is not None, cell_list):
            _, pair_features, _ = self._feature_generator_process([cell], dataset=d)
            
            if all_ones:
                dict_keys = list(pair_features[cell].keys())
                if dict_keys:
                    # 训练样本全为正类时，直接采用第一个候选修复。
                    correction_dict[cell] = dict_keys[0]
            elif all_zeros:
                # 训练样本全为负类时，不对该批单元格输出修复。
                continue
            else:
                dict_keys = list(pair_features[cell].keys())
                if hasattr(classification_model, "decision_function") and self.USE_PREDICTION_CONFIDENCE:
                    # 支持置信度的分类器优先选择分数最高的正类候选。
                    decision_scores = classification_model.decision_function(list(pair_features[cell].values()))
                    positive_indices = numpy.where(decision_scores > 0)[0]
                    if len(positive_indices) > 0:
                        best_index = positive_indices[numpy.argmax(decision_scores[positive_indices])]
                        correction_dict[cell] = dict_keys[best_index]
                else:
                    # 不支持 decision_function 的分类器使用 predict 结果，取第一个正类候选。
                    predictions = classification_model.predict(list(pair_features[cell].values()))
                    for index, predicted_label in enumerate(predictions):
                        if predicted_label:
                            correction_dict[cell] = dict_keys[index]
                            break

        return correction_dict
                

    def predict_correction_multicore(self, classification_model, used_cells_test, d, all_zeros, all_ones):
        """
        使用多进程预测待修复错误单元格。

        输入：
            classification_model：当前列训练得到的候选分类器。
            used_cells_test：待预测错误单元格列表。
            d：Dataset 对象。
            all_zeros：训练候选是否全为负类。
            all_ones：训练候选是否全为正类。
        """
        pool = Pool(self.NUM_WORKERS,initargs=(d,classification_model), initializer=worker_init_prediction)

        prediction_args_generator = itertools.zip_longest(*[iter(used_cells_test)] * self.CHUNK_SIZE)

        correction_iterator = pool.imap(functools.partial(self._prediction_process, all_zeros=all_zeros, all_ones=all_ones), prediction_args_generator)

        for i, correction_dict in enumerate(correction_iterator):
            d.corrected_cells.update(correction_dict)
            if self.VERBOSE:
                print(f"{i*self.CHUNK_SIZE}/{len(used_cells_test)} predicted correction", end="\r")

        pool.close()

    def predict_corrections(self, d):
        """
        为每个已检测错误预测修复值。

        输入：
            d：包含候选模型、已标注单元格和 detected_cells 的 Dataset 对象。

        输出：
            无直接返回值，预测结果写入 d.corrected_cells。
        """
        if self.VERBOSE:
            print("Predicting module...")

        len_column_errors = len(d.column_errors)
        for column_idx, j in enumerate(d.column_errors):
            if self.VERBOSE:
                print("------------------------------------------------------------------------")
                print(f"{column_idx+1}/{len_column_errors} columns({d.dataframe.columns[j]})")
            
            used_cells_train = []
            used_cells_test = []
            for k, cell in enumerate(d.column_errors[j]):
                if cell in d.detected_cells:
                    if cell in d.labeled_cells and d.labeled_cells[cell][0] == 1:
                        # 已标注为错误的单元格用于训练候选修复分类器。
                        used_cells_train.append(cell)
                    else:
                        # 未标注或非训练错误单元格进入预测集合。
                        used_cells_test.append(cell)

            x_train = []
            y_train = []
            
            if self.VERBOSE:
                print(f"Generating train features({len(used_cells_train)}) ...")

            len_used_cells_train = len(used_cells_train)
            for k, (pair_features, cells) in enumerate(self.generate_features(d, used_cells_train)):
                if self.VERBOSE:
                    print(f"{(k)*self.CHUNK_SIZE}/{len_used_cells_train} creating train features",end="\r")
                for cell in cells:
                    for correction, value in pair_features[cell].items():
                        x_train.append(value)
                        y_train.append(int(correction == d.labeled_cells[cell][1]))
                        # 用户已确认的修复值直接写入最终结果。
                        d.corrected_cells[cell] = d.labeled_cells[cell][1]
        
            if x_train:
                if self.VERBOSE:
                    print("Training classifier ...") 
                # 按配置选择候选修复分类器；ABC 是默认配置，适合小样本启动。
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

                all_zeros = False
                all_ones = False
                if sum(y_train) == 0:
                    # 没有正例时跳过预测，避免输出无依据修复。
                    all_zeros = True
                elif sum(y_train) == len(y_train):
                    # 候选全为正例时无需训练分类器，预测阶段会直接选候选。
                    all_ones = True
                else:
                    classification_model.fit(x_train, y_train)

                if self.VERBOSE:
                    print("Predicting corrections...")

                self.predict_correction_multicore(classification_model, used_cells_test, d, all_zeros, all_ones)

        if self.VERBOSE:
            print("{:.0f}% ({} / {}) of data errors are corrected.".format(100 * len(d.corrected_cells) / len(d.detected_cells),
                                                                           len(d.corrected_cells), len(d.detected_cells)))

    def store_results(self, d):
        """
        保存修复流程结果。

        输入：
            d：已完成修复流程的 Dataset 对象。

        输出：
            无直接返回值，会写入 error-correction/correction.dataset 文件。
        """
        ec_folder_path = os.path.join(d.results_folder, "error-correction")
        if not os.path.exists(ec_folder_path):
            os.mkdir(ec_folder_path)
        # 保存完整 Dataset，便于后续分析模型、标注状态和修复结果。
        pickle.dump(d, open(os.path.join(ec_folder_path, "correction.dataset"), "wb"))
        if self.VERBOSE:
            print("The results are stored in {}.".format(os.path.join(ec_folder_path, "correction.dataset")))

    def run(self, d):
        """
        对输入数据集执行完整 Baran 错误修复流程。

        输入：
            d：Dataset 对象，必须包含 detected_cells。

        输出：
            corrected_cells：错误单元格到修复值的字典。

        示例：
            data = raha.dataset.Dataset(dataset_dictionary)
            data.detected_cells = detector.run(dataset_dictionary)
            app = Correction()
            app.LABELING_BUDGET = 10
            app.CLASSIFICATION_MODEL = "ABC"
            corrected_cells = app.run(data)
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "---------------------Initialize the Dataset Object----------------------\n"
                  "------------------------------------------------------------------------")
        d = self.initialize_dataset(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------Initialize Error Corrector Models-------------------\n"
                  "------------------------------------------------------------------------")
        self.initialize_models(d)
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------Iterative Tuple Sampling, Labeling, and Learning----------\n"
                  "------------------------------------------------------------------------")
        while len(d.labeled_tuples) < self.LABELING_BUDGET:
            if self.VERBOSE:
                print(f"Label round {len(d.labeled_tuples)+1}/{self.LABELING_BUDGET}")
            self.sample_tuple(d)
            if d.has_ground_truth:
                # 有 clean_path 时自动标注，适合 benchmark 和回归测试。
                self.label_with_ground_truth(d)
            # 否则：
            #   没有真实标签时，需要按 Jupyter notebook 中的交互流程由用户标注当前元组。
            self.update_models(d)
            self.predict_corrections(d)
            if self.VERBOSE:
                print("------------------------------------------------------------------------")
        if self.SAVE_RESULTS:
            if self.VERBOSE:
                print("------------------------------------------------------------------------\n"
                      "---------------------------Storing the Results--------------------------\n"
                      "------------------------------------------------------------------------")
            self.store_results(d)
        return d.corrected_cells
########################################


########################################
if __name__ == "__main__":
    # 命令行直接运行本文件时，使用内置 flights 数据集和真实错误集合演示修复与评估流程。
    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    data = raha.dataset.Dataset(dataset_dictionary)
    data.detected_cells = dict(data.get_actual_errors_dictionary())
    app = Correction()
    correction_dictionary = app.run(data)
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(data.name, p, r, f))
    # --------------------离线预训练示例--------------------
    # 先从 Wikipedia 修订历史中抽取修复样本，再预训练 value-based 模型。
    # app.extract_revisions(wikipedia_dumps_folder="../wikipedia-data")
    # app.pretrain_value_based_models(revision_data_folder="../wikipedia-data/revision-data")
########################################
