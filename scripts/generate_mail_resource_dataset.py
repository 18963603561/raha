"""
从真实邮件资源 XLS 样本生成 Raha 可用的 clean.csv 与 dirty.csv。

该脚本只抽取用户指定的关键字段，先将真实表中的有效值规范化为干净样本，
再按目标错误率注入常见人工录入错误，便于后续错误检测和修复评估。
"""

import argparse
import ipaddress
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd


# 用户明确指定的关键字段；原需求写 20 个字段，但实际列名清单为 17 个。
SELECTED_COLUMNS = [
    "源IP",
    "源IPv6",
    "上网认证帐号",
    "上网认证类型",
    "域名",
    "IMEI",
    "IMSI",
    "用户ID",
    "用户名",
    "邮件中发件人信息",
    "邮件中收件人信息",
    "数据来源",
    "截获时间",
    "关联纬度",
    "关联经度",
    "基站编号",
    "基站类型",
]

# 日常录入中常见的空值写法，需要统一视为空。
EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "无", "空"}

# 邮箱、域名和坐标的基础校验规则，兼顾真实业务值与样本生成值。
EMAIL_PATTERN = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")
DOMAIN_PATTERN = re.compile(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")
INTEGER_PATTERN = re.compile(r"^\d+$")


def configure_logging() -> None:
    """初始化脚本日志格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def normalize_cell(value: object) -> str:
    """
    规范化单元格文本。

    输入：
        value：从 Excel 读取到的原始值。

    输出：
        去除首尾空白后的字符串；常见空值写法会返回空字符串。
    """
    text = "" if value is None else str(value).strip()
    if text.lower() in EMPTY_TOKENS:
        return ""
    return text


def is_ipv4(value: str) -> bool:
    """判断文本是否为合法 IPv4 地址。"""
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False


def is_ipv6(value: str) -> bool:
    """判断文本是否为合法 IPv6 地址。"""
    try:
        ipaddress.IPv6Address(value)
        return True
    except ValueError:
        return False


def is_latitude(value: str) -> bool:
    """判断文本是否为合法纬度。"""
    try:
        number = float(value)
        return -90 <= number <= 90
    except ValueError:
        return False


def is_longitude(value: str) -> bool:
    """判断文本是否为合法经度。"""
    try:
        number = float(value)
        return -180 <= number <= 180
    except ValueError:
        return False


def luhn_check_digit(number_without_digit: str) -> str:
    """
    计算 IMEI 使用的 Luhn 校验位。

    输入：
        number_without_digit：不含最后一位校验位的数字串。

    输出：
        单字符校验位。
    """
    total = 0
    reversed_digits = list(map(int, reversed(number_without_digit)))
    for index, digit in enumerate(reversed_digits, start=1):
        # IMEI 校验位计算要求从右向左隔位翻倍。
        if index % 2 == 1:
            doubled = digit * 2
            total += doubled // 10 + doubled % 10
        else:
            total += digit
    return str((10 - total % 10) % 10)


def build_imei(row_index: int) -> str:
    """根据行号生成合法 15 位 IMEI。"""
    body = f"860000{row_index + 1:08d}"[:14]
    return body + luhn_check_digit(body)


def build_clean_value(column: str, row_index: int, original_value: str) -> str:
    """
    为指定字段生成干净值。

    优先保留原始表中合法非空值；原始值缺失或格式不适合测试时，按字段类型补齐稳定样本。
    """
    value = normalize_cell(original_value)
    if column == "源IP":
        return value if is_ipv4(value) else f"10.{row_index % 200}.{(row_index * 7) % 250}.{(row_index * 13) % 250 + 1}"
    if column == "源IPv6":
        return value if is_ipv6(value) else f"2409:8000:{row_index % 65535:x}:{(row_index * 3) % 65535:x}::1"
    if column == "上网认证帐号":
        return value if value else f"auth_{500000 + row_index}@access.example.cn"
    if column == "上网认证类型":
        return value if value else str([1020001, 1020002, 1020003, 1020004][row_index % 4])
    if column == "域名":
        return value if DOMAIN_PATTERN.match(value) else f"mail{row_index % 20}.example.cn"
    if column == "IMEI":
        return value if INTEGER_PATTERN.match(value) and len(value) == 15 else build_imei(row_index)
    if column == "IMSI":
        return value if INTEGER_PATTERN.match(value) and len(value) == 15 else f"4600{row_index % 10}{row_index + 1000000000:010d}"[:15]
    if column == "用户ID":
        return value if value else f"uid_{100000 + row_index}"
    if column == "用户名":
        return value if value else f"user_{row_index + 1:06d}"
    if column == "邮件中发件人信息":
        return value if EMAIL_PATTERN.match(value) else f"sender{row_index + 1:06d}@example.cn"
    if column == "邮件中收件人信息":
        return value if EMAIL_PATTERN.match(value) else f"receiver{row_index + 1:06d}@example.cn"
    if column == "数据来源":
        return value if INTEGER_PATTERN.match(value) else str([111, 124, 144, 802][row_index % 4])
    if column == "截获时间":
        return value if INTEGER_PATTERN.match(value) and 10 <= len(value) <= 13 else str(1773500000 + row_index * 37)
    if column == "关联纬度":
        return value if is_latitude(value) else f"{29.400000 + (row_index % 500) / 10000:.6f}"
    if column == "关联经度":
        return value if is_longitude(value) else f"{106.400000 + (row_index % 500) / 10000:.6f}"
    if column == "基站编号":
        return value if value else f"BS500000{row_index + 1:06d}"
    if column == "基站类型":
        return value if value else str([1, 2, 3, 4, 5][row_index % 5])
    return value


def corrupt_value(column: str, clean_value: str, row_index: int, randomizer: random.Random) -> tuple[str, str]:
    """
    按字段类型制造一个脏值。

    输出：
        二元组，分别为脏值和错误类型。
    """
    operation = randomizer.choice(["missing", "typo", "format", "wrong_value"])
    if operation == "missing":
        return "", "漏写"

    if column == "源IP":
        if operation == "format":
            return clean_value.replace(".", "-"), "格式错误"
        if operation == "wrong_value":
            return "999.999.999.999", "值域错误"
        return clean_value[:-1], "少写字符"
    if column == "源IPv6":
        if operation == "format":
            return clean_value.replace(":", ":::", 1), "格式错误"
        if operation == "wrong_value":
            return "2409:::bad-ipv6", "值域错误"
        return clean_value.replace(":", "", 1), "少写分隔符"
    if column in {"IMEI", "IMSI", "截获时间"}:
        if operation == "format":
            return f"{clean_value}x", "混入字母"
        if operation == "wrong_value":
            return "0" * max(1, len(clean_value) - 2), "长度错误"
        return clean_value[:-1], "少写数字"
    if column in {"关联纬度", "关联经度"}:
        if operation == "format":
            return clean_value.replace(".", ",", 1), "小数点误写"
        if operation == "wrong_value":
            return "999.999999", "值域错误"
        return clean_value[:-2], "精度缺失"
    if column in {"邮件中发件人信息", "邮件中收件人信息"}:
        if operation == "format":
            return clean_value.replace("@", " at "), "邮箱格式错误"
        if operation == "wrong_value":
            return "unknown@example", "域名不完整"
        return clean_value.replace(".", "", 1), "少写符号"
    if column == "域名":
        if operation == "format":
            return clean_value.replace(".", "。", 1), "中英文符号混用"
        if operation == "wrong_value":
            return "localhost", "域名不完整"
        return clean_value.replace(".", "", 1), "少写符号"
    if column in {"上网认证帐号", "用户ID", "用户名", "基站编号"}:
        if operation == "format":
            return clean_value.replace("_", " ", 1) if "_" in clean_value else f"{clean_value} wrong", "分隔符误写"
        if operation == "wrong_value":
            return f"{clean_value}_wrong", "写错内容"
        return clean_value[:-1], "少写字符"
    if column in {"上网认证类型", "数据来源", "基站类型"}:
        if operation == "format":
            return f"{clean_value}.0", "数字格式错误"
        if operation == "wrong_value":
            return "未知", "值域错误"
        return clean_value + clean_value[-1:], "重复录入"
    return f"{clean_value}_wrong_{row_index}", "写错内容"


def locate_source_xls(project_root: Path, source_path: str | None) -> Path:
    """
    定位真实 XLS 文件。

    输入：
        project_root：项目根目录。
        source_path：用户可选传入的源文件路径。
    """
    if source_path:
        path = Path(source_path)
        return path if path.is_absolute() else project_root / path
    candidates = sorted((project_root / "doc" / "20260709").glob("*.xls"))
    if not candidates:
        raise FileNotFoundError("未在 doc/20260709 下找到 xls 源文件。")
    return candidates[0]


def build_dataset(
    project_root: Path,
    source_path: Path,
    output_name: str,
    error_rate: float,
    seed: int,
    limit: int | None,
    real_only: bool,
) -> dict:
    """
    构造 clean、dirty 和错误清单文件。

    输出：
        包含数据集规模、错误率和文件路径的摘要字典。
    """
    logging.info("开始读取 XLS 源文件：%s", source_path)
    raw = pd.read_excel(source_path, sheet_name=0, header=None, dtype=str, engine="xlrd")
    chinese_headers = [normalize_cell(value) for value in raw.iloc[0].tolist()]
    data_frame = raw.iloc[2:].copy()
    data_frame.columns = chinese_headers

    missing_columns = [column for column in SELECTED_COLUMNS if column not in data_frame.columns]
    if missing_columns:
        raise ValueError(f"源文件缺少字段：{missing_columns}")

    if limit:
        # 控制样本行数，避免 Raha 测试阶段因为数据过大而耗时过长。
        data_frame = data_frame.head(limit)

    clean_frame = pd.DataFrame()
    for column in SELECTED_COLUMNS:
        if real_only:
            # 真实数据模式不做序列补值，源表为空就保留为空，确保 clean.csv 完全来自 XLS。
            clean_frame[column] = [normalize_cell(value) for value in data_frame[column].tolist()]
        else:
            clean_frame[column] = [
                build_clean_value(column, row_index, value)
                for row_index, value in enumerate(data_frame[column].tolist())
            ]

    dirty_frame = clean_frame.copy()
    randomizer = random.Random(seed)
    total_cells = clean_frame.shape[0] * clean_frame.shape[1]
    target_error_count = round(total_cells * error_rate)
    candidate_cells = []
    for row_index in range(clean_frame.shape[0]):
        for column in SELECTED_COLUMNS:
            # 真实数据模式只污染原本非空的真实值，避免在 clean 为空的字段里凭空生成业务值。
            if (not real_only) or clean_frame.at[row_index, column] != "":
                candidate_cells.append((row_index, column))
    if target_error_count > len(candidate_cells):
        if real_only:
            logging.warning(
                "真实非空单元格不足，按全部真实非空单元格造脏：目标 %s 个，实际 %s 个。",
                target_error_count,
                len(candidate_cells),
            )
            target_error_count = len(candidate_cells)
        else:
            raise ValueError(
                f"可造脏的真实非空单元格不足：需要 {target_error_count} 个，实际只有 {len(candidate_cells)} 个。"
            )
    selected_cells = randomizer.sample(candidate_cells, target_error_count)

    manifest_rows = []
    for row_index, column in selected_cells:
        clean_value = clean_frame.at[row_index, column]
        dirty_value, error_type = corrupt_value(column, clean_value, row_index, randomizer)
        if dirty_value == clean_value:
            # 极少数字段在截断后可能没有变化，此时强制置空以保证真实错误率稳定。
            dirty_value = ""
            error_type = "漏写"
        dirty_frame.at[row_index, column] = dirty_value
        manifest_rows.append(
            {
                "row_index": row_index,
                "column": column,
                "clean_value": clean_value,
                "dirty_value": dirty_value,
                "error_type": error_type,
            }
        )

    output_dir = project_root / "datasets" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean.csv"
    dirty_path = output_dir / "dirty.csv"
    manifest_path = output_dir / "dirty_error_manifest.csv"
    profile_path = output_dir / "dataset_profile.json"

    logging.info("写出 clean.csv：%s", clean_path)
    clean_frame.to_csv(clean_path, index=False, encoding="utf-8", lineterminator="\n")
    logging.info("写出 dirty.csv：%s", dirty_path)
    dirty_frame.to_csv(dirty_path, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(manifest_rows).sort_values(["row_index", "column"]).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    summary = {
        "dataset_name": output_name,
        "source_path": str(source_path),
        "rows": int(clean_frame.shape[0]),
        "columns": int(clean_frame.shape[1]),
        "target_error_rate": error_rate,
        "actual_error_cells": int(target_error_count),
        "actual_error_rate": round(target_error_count / total_cells, 6),
        "real_only": real_only,
        "candidate_non_empty_cells": len(candidate_cells),
        "selected_columns": SELECTED_COLUMNS,
        "clean_path": str(clean_path),
        "dirty_path": str(dirty_path),
        "manifest_path": str(manifest_path),
    }
    profile_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("数据集生成完成：%s", summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成邮件资源 clean/dirty 测试数据集。")
    parser.add_argument("--source", default=None, help="XLS 源文件路径，默认读取 doc/20260709 下的 xls。")
    parser.add_argument("--output-name", default="mail_resource", help="datasets 下的数据集目录名。")
    parser.add_argument("--error-rate", type=float, default=0.40, help="dirty.csv 目标单元格错误率。")
    parser.add_argument("--seed", type=int, default=20260709, help="随机种子，保证造脏结果可复现。")
    parser.add_argument("--limit", type=int, default=None, help="可选样本行数上限。")
    parser.add_argument("--real-only", action="store_true", help="只使用 XLS 真实值，不对 clean.csv 做序列补值。")
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    args = parse_arguments()
    if not 0 < args.error_rate < 1:
        raise ValueError("错误率必须在 0 到 1 之间。")
    source_path = locate_source_xls(project_root, args.source)
    build_dataset(project_root, source_path, args.output_name, args.error_rate, args.seed, args.limit, args.real_only)


if __name__ == "__main__":
    main()
