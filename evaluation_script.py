"""
    使用说明：
        相关第三方库依赖请自行安装。
        测试脚本所在目录需要涵盖共两个文件(json格式)：提交答案文件、参考答案文件。
        只参加部分赛道的评测者注释掉其他赛道的对应部分。

        然后你只需要更改两个部分：
        answer_file_name（提交答案文件名）
        reference_file_name（参考答案文件名）

        运行该脚本，即可开始评估。

    使用样例：
        answer_file_name = "answer_example.json"
        reference_file_name = "reference_example.json"
"""

import os
import json
import copy
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def component_micro_f1_calculate(true_labels_list, pred_labels_list):
    """
    计算Micro F1分数，考虑重叠实体

    参数:
    true_labels_list: 真实标签列表
    pred_labels_list: 预测标签列表

    返回:
    f1: f1分数
    """
    if true_labels_list == [[]] and pred_labels_list == [[]]:
        return 1.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
        # 将标签转换为集合以便比较
        true_set = set()
        pred_set = set()

        # 处理真实标签
        for label in true_labels:
            if label['start'] and label['end']:  # 确保start和end不为空
                # 将每个实体的位置信息转换为元组以便比较
                starts = tuple(label['start'])
                ends = tuple(label['end'])
                true_set.add((label['label'], starts, ends))

        # 处理预测标签
        for label in pred_labels:
            if label['start'] and label['end']:  # 确保start和end不为空
                starts = tuple(label['start'])
                ends = tuple(label['end'])
                pred_set.add((label['label'], starts, ends))

        # 计算TP, FP, FN
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 计算precision, recall, f1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


class LabelOneHotEncoder:
    def __init__(self):
        self.__form_map_categories = {
            '比喻': ['明喻', '暗喻', '借喻'],
            '比拟': ['名词', '动词', '形容词'],
            '夸张': ['直接夸张', '间接夸张'],
            '排比': ['成分排比', '句子排比'],
            '反复': ['间隔反复', '连续反复'],
            '设问': ['问答连属', '问答不连属'],
            '反问': ['单句反问', '复句反问'],
            '摹状': ['通感', '直感']
        }
        self.__content_map_categories = {
            '比喻': ['实在物', '动作', '抽象概念'],
            '比拟': ['拟人', '拟物'],
            '夸张': ['扩大夸张', '缩小夸张', '超前夸张'],
            '排比': ['并列', '承接', '递进']
        }

        # 形式赛道，粗粒度标签独热编码器
        form_track_rhetoric_labels = [i for i in self.__form_map_categories.keys()]
        self.form_track_rhetoric_one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            categories=[form_track_rhetoric_labels])
        # 形式赛道，形式细粒度标签独热编码器
        form_track_form_labels = sum([i for i in self.__form_map_categories.values()], [])
        self.form_track_form_one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            categories=[form_track_form_labels])
        # 内容赛道，粗粒度标签独热编码器
        content_track_rhetoric_labels = [i for i in self.__content_map_categories.keys()]
        self.content_track_rhetoric_one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            categories=[content_track_rhetoric_labels])
        # 内容赛道，内容细粒度标签独热编码器
        content_track_content_labels = sum([i for i in self.__content_map_categories.values()], [])
        self.content_track_content_one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            categories=[content_track_content_labels])

    def __call__(self, labels: list[tuple], track_type: str):
        """
            :parameter labels: 每一个修辞实体的标签都是元组形式(粗粒度类别,细粒度类别,修辞成分(如有))。
            :parameter track_type: 赛道名称, 备选值['form', 'content', 'component']。
        """""
        if track_type == 'form':
            rhetoric_label = np.array([label[0] for label in labels]).reshape(-1, 1)
            rhetoric_label_one_hot = self.form_track_rhetoric_one_hot_encoder.fit_transform(rhetoric_label)
            rhetoric_label_one_hot = np.where(np.sum(rhetoric_label_one_hot, axis=0) > 1,
                                              1, np.sum(rhetoric_label_one_hot, axis=0))

            form_label = np.array([label[1] for label in labels]).reshape(-1, 1)
            form_label_one_hot = self.form_track_form_one_hot_encoder.fit_transform(form_label)
            form_label_one_hot = np.where(np.sum(form_label_one_hot, axis=0) > 1,
                                          1, np.sum(form_label_one_hot, axis=0))
            return rhetoric_label_one_hot, form_label_one_hot

        elif track_type == 'content':
            rhetoric_label = np.array([label[0] for label in labels]).reshape(-1, 1)
            rhetoric_label_one_hot = self.form_track_rhetoric_one_hot_encoder.fit_transform(rhetoric_label)
            rhetoric_label_one_hot = np.where(np.sum(rhetoric_label_one_hot, axis=0) > 1,
                                              1, np.sum(rhetoric_label_one_hot, axis=0))

            content_label = np.array([label[1] for label in labels]).reshape(-1, 1)
            content_label_one_hot = self.content_track_content_one_hot_encoder.fit_transform(content_label)
            content_label_one_hot = np.where(np.sum(content_label_one_hot, axis=0) > 1,
                                             1, np.sum(content_label_one_hot, axis=0))
            return rhetoric_label_one_hot, content_label_one_hot


def read_json_file(file_path):
    if not file_path.endswith('.json'):
        return False
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    return data


class AnswerEvaluator:
    """
        测试阶段答案评估类。
        :parameter answer_data_path: 指定你的模型输出文件路径，即你需要提交的答案json文件路径。
        :parameter reference_data_path: 指定你的参考答案文件路径（如有）。
        :parameter track_type: 指定测试赛道。备选值：["form","content","component"]。
    """

    def __init__(
            self,
            answer_data_path: str,
            reference_data_path: str,
            track_type: str = None
    ):
        self.answer_data = read_json_file(answer_data_path)
        self.reference_data = read_json_file(reference_data_path)
        self.track_type = track_type
        self.one_hot_encoder = LabelOneHotEncoder()

    @staticmethod
    def get_plain_idx_list(total_sentence_num, rhetorical_sentence_idx_list):
        """ 获取无修辞的句子下标 """
        plain_list = []
        for sentence_idx in range(total_sentence_num):
            if sentence_idx not in rhetorical_sentence_idx_list and sentence_idx not in plain_list:
                plain_list.append(sentence_idx)
        return plain_list

    def remove_duplicate_rhetoric(self):
        """ 去除重复的修辞实体识别结果"""
        for i, _ in enumerate(self.reference_data):
            for j, _ in enumerate(self.reference_data[i]['rhetoricItems']):
                rhetoricList = copy.deepcopy(self.reference_data[i]['rhetoricItems'][j]['rhetoricList'])
                rhetoricList_temp = []
                for k in rhetoricList:
                    if k not in rhetoricList_temp:
                        rhetoricList_temp.append(k)
                self.reference_data[i]['rhetoricItems'][j]['rhetoricList'] = rhetoricList_temp

        for i, _ in enumerate(self.answer_data):
            for j, _ in enumerate(self.answer_data[i]['rhetoricItems']):
                rhetoricList = copy.deepcopy(self.answer_data[i]['rhetoricItems'][j]['rhetoricList'])
                rhetoricList_temp = []
                for k in rhetoricList:
                    if k not in rhetoricList_temp:
                        rhetoricList_temp.append(k)
                self.answer_data[i]['rhetoricItems'][j]['rhetoricList'] = rhetoricList_temp

    def check_keys(self):
        """ 检查提交答案的数据格式（字典键） """
        item_keys = ['idx', 'document', 'sentenceList', 'rhetoricItems']
        for item in self.answer_data:
            for item_key in item_keys:
                # 检查第一级字典键完备（数据项）
                if item_key not in item:
                    return False
            rhetoricItem_keys = ['sentenceIdx', 'rhetoricList']
            for rhetoricItem in item['rhetoricItems']:
                for rhetoricItem_key in rhetoricItem_keys:
                    # 检查第二级字典键完备（修辞句组项）
                    if rhetoricItem_key not in rhetoricItem:
                        return False
                rhetoric_keys = ['rhetoric']
                if self.track_type == 'form':
                    rhetoric_keys += ['form']
                elif self.track_type == 'content':
                    rhetoric_keys += ['content']
                elif self.track_type == 'component':
                    rhetoric_keys += ['form',
                                      'conjunction', 'conjunctionBeginIdx', 'conjunctionEndIdx',
                                      'tenor', 'tenorBeginIdx', 'tenorEndIdx',
                                      'vehicle', 'vehicleBeginIdx', 'vehicleEndIdx']
                for rhetoric in rhetoricItem['rhetoricList']:
                    for rhetoric_key in rhetoric_keys:
                        # 检查第三级字典键完备（修辞实体项）
                        if rhetoric_key not in rhetoric:
                            return False
        else:
            return True

    def check_idx(self):
        """ 检查提交答案中各 document 的 idx 是否唯一且完全 """
        answer_idx_list = [item['idx'] for item in self.answer_data]
        reference_idx_list = [item['idx'] for item in self.reference_data]
        if len(answer_idx_list) != len(set(answer_idx_list)) or len(reference_idx_list) != len(set(reference_idx_list)):
            return False
        if set(answer_idx_list) != set(reference_idx_list):
            return False
        return True

    @staticmethod
    def component_score_calculate(
            reference_label_list,
            answer_label_list,
    ):
        ref_conjunctions = [item[0] for item in reference_label_list if item[0]['start'] and item[0]['end']]
        ref_tenors = [item[1] for item in reference_label_list if item[1]['start'] and item[1]['end']]
        ref_vehicles = [item[2] for item in reference_label_list if item[2]['start'] and item[2]['end']]

        ans_conjunctions = [item[0] for item in answer_label_list if item[0]['start'] and item[0]['end']]
        ans_tenors = [item[1] for item in answer_label_list if item[1]['start'] and item[1]['end']]
        ans_vehicles = [item[2] for item in answer_label_list if item[2]['start'] and item[2]['end']]

        conjunction_f1_score = component_micro_f1_calculate([ref_conjunctions], [ans_conjunctions])
        tenor_f1_score = component_micro_f1_calculate([ref_tenors], [ans_tenors])
        vehicle_f1_score = component_micro_f1_calculate([ref_vehicles], [ans_vehicles])

        f1_score_dict = {
            'conjunction': conjunction_f1_score,
            'tenor': tenor_f1_score,
            'vehicle': vehicle_f1_score
        }
        return f1_score_dict

    @staticmethod
    def location_score_calculate(
            answer_rhetorical_list, answer_plain_list,
            reference_rhetorical_list, reference_plain_list
    ):
        # 去除掉提交答案中重复出现的修辞句组对修辞定位分数的影响
        answer_rhetorical_list_without_duplicate = []
        for i in answer_rhetorical_list:
            if i not in answer_rhetorical_list_without_duplicate:
                answer_rhetorical_list_without_duplicate.append(i)
        answer_rhetorical_list = answer_rhetorical_list_without_duplicate

        # 无修辞 document 判断正确，得满分
        if not answer_rhetorical_list and not reference_rhetorical_list:
            return 1.0
        # 无修辞 document 判断错误，得 0 分
        elif not reference_rhetorical_list and answer_rhetorical_list:
            return 0.0

        # 对于朴素句，直接整体计算 IoU。我们认为没有无修辞句组时，朴素句的定位得分为 1
        if not set(answer_plain_list) | set(reference_plain_list):
            plain_score = 1.0
        else:
            plain_score = (len(set(answer_plain_list) & set(reference_plain_list)) /
                           len(set(answer_plain_list) | set(reference_plain_list)))

        # 对于修辞句，对于每一条答案中的修辞句组，我们计算命中的（交集不为空）句组与对应参考句组的 IoU（取最大）。
        # 对于所有命中的句组，我们取 IoU 的平均结果。
        iou_list = []
        for ans_sentenceList in answer_rhetorical_list:
            hit_iou_list = []
            for ref_sentenceList in reference_rhetorical_list:
                if set(ans_sentenceList) & set(ref_sentenceList):
                    hit_iou_list.append(
                        len(set(ans_sentenceList) & set(ref_sentenceList)) /
                        len(set(ans_sentenceList) | set(ref_sentenceList))
                    )
            else:
                if hit_iou_list:
                    iou_list.append(max(hit_iou_list))
        rhetorical_score = sum(iou_list) / len(reference_rhetorical_list)

        return (plain_score + rhetorical_score) / 2

    def rhetoric_score_calculate(
            self,
            answer_tuple_list: list[tuple],
            reference_tuple_list: list[tuple],
            track_type: str
    ):
        rhetorical_score_list = []  # 用于记录无交集，或有交集但对应参考句组为“无修辞”时的f1 score。
        track_rhetorical_score = {
            "rhetoric_score_list": [],
            "form_score_list": [],
            "content_score_list": [],
            "conjunction_score_list": [],
            "tenor_score_list": [],
            "vehicle_score_list": []
        }  # 用于记录各赛道修辞评分结果。

        # 如果参考答案中所有句子无修辞，且提交结果为空，则算满分
        if (not [i for i in reference_tuple_list if i[1]]) and (not answer_tuple_list):
            return 1.0
        for ref_sentenceIdx, ref_rhetoricList in reference_tuple_list:
            ans_rhetoric4ref = []
            for ans_sentenceIdx, ans_rhetoricList in answer_tuple_list:
                # 有交集即计算修辞识别情况，因此需要成分抽取赛道中成分的 idx 以 document 全局下标结果为准。
                if set(ref_sentenceIdx) & set(ans_sentenceIdx):
                    ans_rhetoric4ref += ans_rhetoricList
            # 去重
            ans_rhetoric4ref_final = []
            for rhetoric in ans_rhetoric4ref:
                if rhetoric not in ans_rhetoric4ref_final:
                    ans_rhetoric4ref_final.append(rhetoric)

            # 如果无交集，且该参考句组为“有修辞”，则 f1 score = 0。
            if not ans_rhetoric4ref_final and ref_rhetoricList:
                rhetorical_score_list.append(0.0)
                continue
            # 如果无交集，且该参考句组为“无修辞”，则不算分。
            if not ans_rhetoric4ref_final and not ref_rhetoricList:
                continue
            # 如果有交集，且该参考句组为“无修辞”，则 f1 score = 0。
            if ans_rhetoric4ref_final and not ref_rhetoricList:
                rhetorical_score_list.append(0.0)
                continue
            # 如果有交集，且该参考句组为“有修辞”，则计算 f1 score。
            # 形式细粒度赛道
            if track_type == 'form':
                form_track_result_ref = [
                    (rhetoric['rhetoric'], rhetoric['form']) for rhetoric in ref_rhetoricList]
                form_track_result_ref = self.one_hot_encoder(labels=form_track_result_ref, track_type=track_type)
                form_track_result_ans = [
                    (rhetoric['rhetoric'], rhetoric['form']) for rhetoric in ans_rhetoric4ref_final]
                form_track_result_ans = self.one_hot_encoder(labels=form_track_result_ans, track_type=track_type)

                rhetoric_micro_f1 = f1_score(y_true=[form_track_result_ref[0]],
                                             y_pred=[form_track_result_ans[0]], average='micro', zero_division=0)
                track_rhetorical_score["rhetoric_score_list"].append(rhetoric_micro_f1)

                form_micro_f1 = f1_score(y_true=[form_track_result_ref[1]],
                                         y_pred=[form_track_result_ans[1]], average='micro', zero_division=0)
                track_rhetorical_score["form_score_list"].append(form_micro_f1)

            elif track_type == 'content':
                content_track_result_ref = [
                    (rhetoric['rhetoric'], rhetoric['content']) for rhetoric in ref_rhetoricList]
                content_track_result_ref = self.one_hot_encoder(labels=content_track_result_ref, track_type=track_type)
                content_track_result_ans = [
                    (rhetoric['rhetoric'], rhetoric['content']) for rhetoric in ans_rhetoric4ref_final]
                content_track_result_ans = self.one_hot_encoder(labels=content_track_result_ans, track_type=track_type)

                rhetoric_micro_f1 = f1_score(y_true=[content_track_result_ref[0]],
                                             y_pred=[content_track_result_ans[0]], average='micro', zero_division=0)
                track_rhetorical_score["rhetoric_score_list"].append(rhetoric_micro_f1)

                content_micro_f1 = f1_score(y_true=[content_track_result_ref[1]],
                                            y_pred=[content_track_result_ans[1]], average='micro', zero_division=0)
                track_rhetorical_score["content_score_list"].append(content_micro_f1)

            elif track_type == 'component':
                component_track_result_ref = [
                    ({
                         "label": 'conjunction',
                         "start": rhetoric['conjunctionBeginIdx'],
                         "end": rhetoric['conjunctionEndIdx']
                     }, {
                         "label": 'tenor',
                         "start": rhetoric['tenorBeginIdx'],
                         "end": rhetoric['tenorEndIdx']
                     }, {
                         "label": 'vehicle',
                         "start": rhetoric['vehicleBeginIdx'],
                         "end": rhetoric['vehicleEndIdx']
                     }
                    ) for rhetoric in ref_rhetoricList
                ]
                component_track_result_ans = [
                    ({
                         "label": 'conjunction',
                         "start": rhetoric['conjunctionBeginIdx'],
                         "end": rhetoric['conjunctionEndIdx']
                     }, {
                         "label": 'tenor',
                         "start": rhetoric['tenorBeginIdx'],
                         "end": rhetoric['tenorEndIdx']
                     }, {
                         "label": 'vehicle',
                         "start": rhetoric['vehicleBeginIdx'],
                         "end": rhetoric['vehicleEndIdx']
                     }
                    ) for rhetoric in ans_rhetoric4ref_final
                ]
                component_micro_f1 = self.component_score_calculate(
                    reference_label_list=component_track_result_ref,
                    answer_label_list=component_track_result_ans
                )
                track_rhetorical_score['conjunction_score_list'].append(component_micro_f1['conjunction'])
                track_rhetorical_score['tenor_score_list'].append(component_micro_f1['tenor'])
                track_rhetorical_score['vehicle_score_list'].append(component_micro_f1['vehicle'])

        if track_type == 'form':
            rhetoric_score_list = rhetorical_score_list + track_rhetorical_score["rhetoric_score_list"]
            if rhetoric_score_list:
                track_rhetorical_score["rhetoric_score_list"] = sum(rhetoric_score_list) / len(rhetoric_score_list)
            else:
                track_rhetorical_score["rhetoric_score_list"] = 0.0
            form_score_list = rhetorical_score_list + track_rhetorical_score["form_score_list"]

            if form_score_list:
                track_rhetorical_score["form_score_list"] = sum(form_score_list) / len(form_score_list)
            else:
                track_rhetorical_score["form_score_list"] = 0.0
            rhetorical_score = 0.3 * track_rhetorical_score["rhetoric_score_list"] + 0.7 * track_rhetorical_score[
                "form_score_list"]
            return rhetorical_score

        elif track_type == 'content':
            rhetoric_score_list = rhetorical_score_list + track_rhetorical_score["rhetoric_score_list"]
            if rhetoric_score_list:
                track_rhetorical_score["rhetoric_score_list"] = sum(rhetoric_score_list) / len(rhetoric_score_list)
            else:
                track_rhetorical_score["rhetoric_score_list"] = 0.0

            content_score_list = rhetorical_score_list + track_rhetorical_score["content_score_list"]
            if content_score_list:
                track_rhetorical_score["content_score_list"] = sum(content_score_list) / len(content_score_list)
            else:
                track_rhetorical_score["content_score_list"] = 0.0
            rhetorical_score = 0.3 * track_rhetorical_score["rhetoric_score_list"] + 0.7 * track_rhetorical_score[
                "content_score_list"]
            return rhetorical_score
        elif track_type == 'component':
            conjunction_score_list = rhetorical_score_list + track_rhetorical_score["conjunction_score_list"]
            if conjunction_score_list:
                track_rhetorical_score["conjunction_score_list"] = (sum(conjunction_score_list) /
                                                                    len(conjunction_score_list))
            else:
                track_rhetorical_score["conjunction_score_list"] = 0.0

            tenor_score_list = rhetorical_score_list + track_rhetorical_score["tenor_score_list"]
            if tenor_score_list:
                track_rhetorical_score["tenor_score_list"] = (sum(tenor_score_list) /
                                                              len(tenor_score_list))
            else:
                track_rhetorical_score["tenor_score_list"] = 0.0

            vehicle_score_list = rhetorical_score_list + track_rhetorical_score["vehicle_score_list"]
            if vehicle_score_list:
                track_rhetorical_score["vehicle_score_list"] = (sum(vehicle_score_list) /
                                                                len(vehicle_score_list))
            else:
                track_rhetorical_score["vehicle_score_list"] = 0.0
            rhetorical_score = ((track_rhetorical_score["conjunction_score_list"] / 3) +
                                (track_rhetorical_score["tenor_score_list"] / 3) +
                                (track_rhetorical_score["vehicle_score_list"] / 3))
            return rhetorical_score
        return False

    @staticmethod
    def merge_intersecting_entries(data):
        groups = []
        for indices, attrs in data:
            # 确保属性列表不为None，如果是None则初始化为空列表
            if attrs is None:
                attrs = []
            groups.append((set(indices), attrs))

        # 合并循环，直到没有合并发生
        while True:
            merged_groups = []
            has_merged = False

            for current_indices, current_attrs in groups:
                found = False
                # 检查当前组是否能与已合并的组中的某个合并
                for i in range(len(merged_groups)):
                    existing_indices, existing_attrs = merged_groups[i]
                    # 如果存在交集，则合并
                    if existing_indices & current_indices:
                        new_indices = existing_indices | current_indices
                        # 合并属性列表，确保不为None
                        combined_attrs = (existing_attrs if existing_attrs is not None else []) + \
                                         (current_attrs if current_attrs is not None else [])
                        # 去重属性列表
                        seen = set()
                        unique_attrs = []
                        for d in combined_attrs:
                            # 将字典转换为可哈希的元组形式
                            # t = tuple(sorted(d.items()))
                            t = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(d.items())))
                            if t not in seen:
                                seen.add(t)
                                unique_attrs.append(d)
                        # 更新已合并的组
                        merged_groups[i] = (new_indices, unique_attrs)
                        found = True
                        has_merged = True
                        break
                if not found:
                    merged_groups.append((current_indices.copy(), current_attrs.copy()))

            # 如果没有发生合并，结束循环
            if not has_merged:
                break
            else:
                groups = merged_groups

        # 将结果转换回有序列表形式
        result = []
        for indices, attrs in groups:
            sorted_indices = sorted(indices)
            result.append((sorted_indices, attrs))

        new_result = []
        for item in result:
            if item not in new_result:
                if item[1]:
                    new_result.append(item)
                else:
                    new_result.append((item[0], None))
        return new_result

    @staticmethod
    def component_idx_to_global(rhetoricList, sentence_idx_list, document_sentence_list):
        if not rhetoricList:
            return sentence_idx_list, rhetoricList
        pretext = "".join(document_sentence_list[:sentence_idx_list[0]])
        offset = len(pretext)
        for i, rhetoric in enumerate(rhetoricList):
            rhetoricList[i].update(
                {'conjunctionBeginIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["conjunctionBeginIdx"]]}
            )
            rhetoricList[i].update(
                {'conjunctionEndIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["conjunctionEndIdx"]]}
            )
            rhetoricList[i].update(
                {'tenorBeginIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["tenorBeginIdx"]]}
            )
            rhetoricList[i].update(
                {'tenorEndIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["tenorEndIdx"]]}
            )
            rhetoricList[i].update(
                {'vehicleBeginIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["vehicleBeginIdx"]]}
            )
            rhetoricList[i].update(
                {'vehicleEndIdx': [idx + offset if idx != -1 else idx for idx in rhetoric["vehicleEndIdx"]]}
            )
        return sentence_idx_list, rhetoricList

    def evaluate(self):
        """
            提交答案评估。
            loc_score: 修辞句组定位分数
            rhetoric_score: 修辞识别分数
        """
        final_score = []
        for i, _ in enumerate(self.reference_data):
            print("\"idx\":", self.reference_data[i]['idx'])

            ans_rhetoricItems = self.answer_data[i]['rhetoricItems']
            ref_rhetoricItems = self.reference_data[i]['rhetoricItems']
            # 逐 document 计算修辞句组定位得分（基于 IoU）
            answer_rhetorical_list = [rhetoricItem['sentenceIdx'] for rhetoricItem in ans_rhetoricItems]
            reference_rhetorical_list = [rhetoricItem['sentenceIdx'] for rhetoricItem in ref_rhetoricItems]
            loc_score = self.location_score_calculate(
                answer_rhetorical_list=copy.deepcopy(answer_rhetorical_list),
                answer_plain_list=self.get_plain_idx_list(
                    total_sentence_num=len(self.answer_data[i]['sentenceList']),
                    rhetorical_sentence_idx_list=sum(answer_rhetorical_list, [])),
                reference_rhetorical_list=copy.deepcopy(reference_rhetorical_list),
                reference_plain_list=self.get_plain_idx_list(
                    total_sentence_num=len(self.reference_data[i]['sentenceList']),
                    rhetorical_sentence_idx_list=sum(reference_rhetorical_list, []))
            )
            # 逐 document 计算修辞得分。
            # 我们认为修辞识别比句组定位更为重要，因此只要句组命中（交集不为空）即进行修辞识别结果的 f1 计算
            # 对每个参考答案，我们考虑提交答案中修辞句组命中了的项，其修辞实体识别结果与参考答案之间的 f1 score。
            # 我们要求标出的句组有且仅有“有修辞”这一种情况，未标出的句子默认为“无修辞”类别。
            # 参考答案中“无修辞”的句子参与 f1 计算。提交答案中“无修辞”的句子不参与 f1 计算。
            sentenceIdx_rhetoric_tuple_list = []  # 每一个元组为(句组下标列表,修辞实体列表)
            begin_idx = 0
            end_idx = len(self.reference_data[i]['sentenceList']) - 1
            for ref_rhetoricItem in self.reference_data[i]['rhetoricItems']:
                while (begin_idx <
                       min(ref_rhetoricItem['sentenceIdx']) <= max(ref_rhetoricItem['sentenceIdx'])
                       < end_idx):
                    sentenceIdx_rhetoric_tuple_list.append(([begin_idx], None))
                    begin_idx += 1
                sentenceIdx_rhetoric_tuple_list.append((ref_rhetoricItem['sentenceIdx'],
                                                        ref_rhetoricItem['rhetoricList']))
                begin_idx = max(ref_rhetoricItem['sentenceIdx']) + 1
            else:
                while begin_idx <= end_idx:
                    sentenceIdx_rhetoric_tuple_list.append(([begin_idx], None))
                    begin_idx += 1

            answer_tuple_list = [
                (item['sentenceIdx'], item['rhetoricList']) for item in self.answer_data[i]['rhetoricItems']
            ]
            # 为了切分定位和修辞两个评分方向，将参考答案中下标存在交集的句组合为一个修辞大句组考虑（句组下标取并集）
            if self.track_type == 'form' or self.track_type == 'content':
                sentenceIdx_rhetoric_tuple_list = self.merge_intersecting_entries(sentenceIdx_rhetoric_tuple_list)
            elif self.track_type == 'component':
                # 成分抽取任务时，需先讲 idx 对齐至 document 全局 idx，然后才能进行合并。
                for j, _ in enumerate(sentenceIdx_rhetoric_tuple_list):
                    sentenceIdx_rhetoric_tuple_list[j] = self.component_idx_to_global(
                        rhetoricList=sentenceIdx_rhetoric_tuple_list[j][1],
                        sentence_idx_list=sentenceIdx_rhetoric_tuple_list[j][0],
                        document_sentence_list=self.reference_data[i]['sentenceList']
                    )
                for j, _ in enumerate(answer_tuple_list):
                    answer_tuple_list[j] = self.component_idx_to_global(
                        rhetoricList=answer_tuple_list[j][1],
                        sentence_idx_list=answer_tuple_list[j][0],
                        document_sentence_list=self.reference_data[i]['sentenceList']
                    )
                sentenceIdx_rhetoric_tuple_list = self.merge_intersecting_entries(sentenceIdx_rhetoric_tuple_list)

            rhetoric_score = self.rhetoric_score_calculate(
                answer_tuple_list=answer_tuple_list,
                reference_tuple_list=sentenceIdx_rhetoric_tuple_list,
                track_type=self.track_type
            )
            print('location score:', loc_score)
            print("rhetoric score:", rhetoric_score)
            final_score.append(0.3 * loc_score + 0.7 * rhetoric_score)
        print(f"您提交的结果在赛道：“{self.track_type}” 中的综合得分为：{sum(final_score) / len(final_score)}")
        return sum(final_score) / len(final_score)

    def __call__(self):
        assert self.check_keys(), "提交答案数据格式不正确，请检查各字典的键。"
        assert self.check_idx(), "提交答案数据中各document的idx重复或缺少。"
        # 按下标对齐 idx
        self.answer_data.sort(key=lambda x: x['idx'])
        self.reference_data.sort(key=lambda x: x['idx'])
        # 对于识别任务，一篇 document 相同识别结果保留一个即可，去除识别的重复项
        self.remove_duplicate_rhetoric()
        # 评估
        return self.evaluate()


if __name__ == '__main__':
    answer_file_name = "answer_example.json"
    reference_file_name = "reference_example.json"
    answer_file_path = os.path.join(os.path.dirname(__file__), answer_file_name)
    reference_file_path = os.path.join(os.path.dirname(__file__), reference_file_name)

    score = {
        'form': 0.0,
        'content': 0.0,
        'component': 0.0
    }

    # 赛道 1: 形式细粒度识别
    answer_evaluator = AnswerEvaluator(
        answer_data_path=answer_file_path,
        reference_data_path=reference_file_path,
        track_type='form'
    )
    score['form'] = answer_evaluator()

    # 赛道 2: 形式细粒度识别
    answer_evaluator = AnswerEvaluator(
        answer_data_path=answer_file_path,
        reference_data_path=reference_file_path,
        track_type='content'
    )
    score['content'] = answer_evaluator()

    # 赛道 3: 形式细粒度识别
    answer_evaluator = AnswerEvaluator(
        answer_data_path=answer_file_path,
        reference_data_path=reference_file_path,
        track_type='component'
    )
    score['component'] = answer_evaluator()
    print(score)
