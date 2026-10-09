import time
from typing import Literal

import numpy as np
from PIL import Image
from rapidocr.utils.typings import EngineType, ModelType, OCRVersion

from .assets import AssetOcr
from .coordinate import get_scale, scale_region
from .log import logger
from .point import Point, Rectangle
from .rapid_model import LANG_TYPE, MODEL_TYPE, OCR_VERSION, PADDLE_ENGINE, auto_download
from .screenshot import ScreenShot
from .window import window_manager

DET_LIMIT_SIDE_LEN: int = 736
"""检测输入的最长边。取 `max` 时不会放大小图，避免白白增加检测耗时"""

DET_THRESH: float = 0.3
"""检测二值化阈值"""

DET_BOX_THRESH: float = 0.5
"""检测框置信度阈值"""

DET_UNCLIP_RATIO: float = 1.5
"""检测框扩张系数"""

TEXT_SCORE: float = 0.0
"""识别结果最低置信度，交由 `RuleOcr` 按素材阈值过滤"""

ONNX_INTRA_OP_NUM_THREADS: int = 4
"""ONNX 单算子内并行线程数。

RapidOCR 的 onnx 模型输入维度是动态的，线程数过多时 ORT 的线程同步开销会反噬，
实测线程数 ≥8 时单次识别耗时劣化一个数量级，故固定为 4（低配机器也不会过度并行）。
"""

ONNXRUNTIME_ENGINE_PARAMS: dict = {
    "EngineConfig.onnxruntime.enable_cpu_mem_arena": True,
    "EngineConfig.onnxruntime.intra_op_num_threads": ONNX_INTRA_OP_NUM_THREADS,
}
"""ONNX Runtime 引擎参数"""

PADDLE_ENGINE_PARAMS: dict = {
    "EngineConfig.paddle.use_cuda": True,
    "EngineConfig.paddle.cuda_ep_cfg.device_id": 0,
    "EngineConfig.paddle.cuda_ep_cfg.gpu_mem": 500,
}
"""PaddlePaddle 引擎参数，GPU 版使用 CUDA 加速"""


def build_engine_params() -> dict:
    """按当前引擎组装 RapidOCR 推理参数

    - PaddlePaddle 引擎：Det/Rec 均指定 `EngineType.PADDLE`，并开启 CUDA
    - ONNX Runtime 引擎：使用默认引擎，额外调整内存分配与线程数

    模型选择（`ocr_version` / `model_type` / `lang_type`）必须显式传入，
    否则 RapidOCR 会退回其 `config.yaml` 的默认值，导致与 `rapid_model`
    校验/下载的模型不是同一套。注意 RapidOCR 要求前两个是 Enum，不能传字符串。

    Returns:
        dict: RapidOCR `params`
    """
    from .rapid_model import ENGINE_TYPE

    engine_type = EngineType(ENGINE_TYPE)
    params = {
        "Global.use_det": True,
        "Global.use_cls": False,
        "Global.use_rec": True,
        "Global.text_score": TEXT_SCORE,
        "Det.engine_type": engine_type,
        "Det.ocr_version": OCRVersion(OCR_VERSION),
        "Det.model_type": ModelType(MODEL_TYPE),
        "Det.lang_type": LANG_TYPE,
        "Rec.engine_type": engine_type,
        "Rec.ocr_version": OCRVersion(OCR_VERSION),
        "Rec.model_type": ModelType(MODEL_TYPE),
        "Rec.lang_type": LANG_TYPE,
        "Det.thresh": DET_THRESH,
        "Det.box_thresh": DET_BOX_THRESH,
        "Det.unclip_ratio": DET_UNCLIP_RATIO,
        "Det.limit_type": "max",
        "Det.limit_side_len": DET_LIMIT_SIDE_LEN,
    }

    if ENGINE_TYPE == PADDLE_ENGINE:
        params.update(PADDLE_ENGINE_PARAMS)
    else:
        params.update(ONNXRUNTIME_ENGINE_PARAMS)

    return params


ENGINE_PARAMS: dict = build_engine_params()
"""RapidOCR 推理参数，检测阈值沿用原 PaddleOCR 的 `det_db_*` 取值以保持识别行为一致"""


def check_ocr_folder():
    """检查OCR资源是否存在，不存在则自动下载"""
    return auto_download()


class OCRManager:
    """OCR 引擎管理器

    负责 RapidOCR 的初始化、资源管理和检测调用。
    通过模块级 ocr_manager 实例全局共享，避免重复初始化。
    """

    def __init__(self):
        self.rapidocr = None  # RapidOCR 引擎实例，非 None 表示已初始化
        self.engine_type: str = ""  # 实际使用的引擎，供日志与排障

    def is_initialized(self) -> bool:
        """检查OCR是否已初始化"""
        return self.rapidocr is not None

    def init(self) -> bool:
        """初始化OCR

        引擎按环境自动选择：装有 `paddle` 包（GPU 版）走 PaddlePaddle + CUDA，
        否则走 ONNX Runtime + CPU。

        Returns:
            bool: 初始化成功/已初始化返回 True
        """
        if self.is_initialized():
            return True

        from .rapid_model import ENGINE_TYPE

        try:
            from rapidocr import RapidOCR

            from .application import MODEL_DIR_PATH

            self.engine_type = ENGINE_TYPE
            logger.ui(f"开始初始化文字识别模型[RapidOCR {OCR_VERSION} {MODEL_TYPE} / {ENGINE_TYPE}]")
            t_start = time.perf_counter()
            params = build_engine_params()
            # 模型统一存放在项目的 models 目录下
            params["Global.model_root_dir"] = str(MODEL_DIR_PATH)
            self.rapidocr = RapidOCR(params=params)
            # RapidOCR 惰性加载模型，此处用一张空图触发加载，
            # 让加载耗时与异常都发生在初始化阶段，而不是首次识别时
            self.rapidocr(np.zeros((32, 32, 3), dtype=np.uint8))

            t_end = time.perf_counter()
            logger.ui(f"模型[RapidOCR {ENGINE_TYPE}]初始化成功，用时 {(t_end - t_start):.2f} 秒")
            return True

        except Exception as e:
            logger.error(f"模型[RapidOCR {OCR_VERSION} {MODEL_TYPE} / {ENGINE_TYPE}]初始化失败: {e}")
            raise

    def detect(self, image: Image.Image) -> list:
        """执行OCR检测

        Args:
            image: PIL Image 对象

        Returns:
            list: 格式化后的OCR检测结果，每项含 Text / Score / BoxPoints
        """
        if not self.is_initialized():
            logger.ui_error("模型未初始化成功，请重启后再试")
            return []

        try:
            t1 = time.perf_counter()
            img_np = np.array(image)
            result = self.rapidocr(img_np)
            t2 = time.perf_counter()
            logger.debug(f"OCR总耗时: {(t2 - t1) * 1000:.2f} ms")
            return get_ocrdata_from_result(result)
        except Exception as e:
            logger.error(f"OCR检测失败: {e}")
            return []


ocr_manager = OCRManager()
"""全局OCR管理器实例"""


def get_ocrdata_from_result(result) -> list:
    """从 RapidOCR 原始结果中提取结构化数据

    将 `RapidOCROutput` 的 boxes / txts / scores 转换为统一的中间格式，
    便于后续 OcrData 封装。boxes 为矩形四角坐标 [左上, 右上, 右下, 左下]。

    Args:
        result: RapidOCR 引擎的调用结果

    Returns:
        list[dict]: 格式化后的结果列表，每项含 Text / Score / BoxPoints
    """
    boxes, txts, scores = result.boxes, result.txts, result.scores
    if boxes is None or txts is None or scores is None:
        return []

    result_list = []
    for text, score, box in zip(txts, scores, boxes):
        item = {
            "Text": text,
            "Score": float(score),
            "BoxPoints": [
                {"X": int(box[0][0]), "Y": int(box[0][1])},
                {"X": int(box[2][0]), "Y": int(box[2][1])},
                {"X": int(box[2][0]), "Y": int(box[2][1])},
                {"X": int(box[0][0]), "Y": int(box[0][1])},
            ],
        }
        result_list.append(item)
    return result_list


class OcrData:
    text: str
    """识别文本"""
    score: float
    """分数阈值"""
    rect: Rectangle
    """识别区域"""
    center: Point
    """识别区域中心坐标"""

    def __init__(self, item: dict) -> None:
        """
        Args:
            item: get_ocrdata_from_result 输出的单条识别结果，
                  dict 结构 {"Text": str, "Score": float, "BoxPoints": list[dict{X, Y}]}
        """
        self.score: float = round(item["Score"], 2)
        self.text: str = item["Text"]
        _BoxPoints = item["BoxPoints"]
        self.x1: int = _BoxPoints[0]["X"]
        self.y1: int = _BoxPoints[0]["Y"]
        self.x2: int = _BoxPoints[2]["X"]
        self.y2: int = _BoxPoints[2]["Y"]
        self.rect = Rectangle(self.x1, self.y1, x2=self.x2, y2=self.y2)
        self.center = self.rect.get_center_point()

    def to_reference(self) -> None:
        """实际客户区坐标 → 基准空间坐标（业务侧统一使用基准空间）"""
        fx, fy = get_scale()
        self.x1, self.y1 = self.x1 / fx, self.y1 / fy
        self.x2, self.y2 = self.x2 / fx, self.y2 / fy
        self.rect = Rectangle(self.x1, self.y1, x2=self.x2, y2=self.y2)
        self.center = self.rect.get_center_point()

    def __repr__(self) -> str:
        return f"text: {self.text}, score: {self.score}, rect: {self.rect.get_box()}, center: {self.center}"


class OcrDetector:
    """OCR检测器，负责截图、OCR调用和结果处理"""

    def __init__(self, region: tuple | None = None):
        """
        Args:
            region: 检测区域，格式为 (x, y, width, height)
        """
        self.region = region or window_manager.current.client_rect

    def get_raw_result(self) -> list[OcrData]:
        """执行截图 + OCR 识别，返回结构化结果

        流程：截图 → RapidOCR 检测 → 格式转换 → 过滤无效数据

        Returns:
            list[OcrData]: 过滤后的 OCR 识别结果列表
        """
        if not ocr_manager.is_initialized():
            ocr_manager.init()

        start_time = time.time()
        screenshot = ScreenShot(rect=self.region)
        image = screenshot.get_image()

        ocr_result = ocr_manager.detect(image)
        data_result: list[OcrData] = []
        for item in ocr_result:
            # 过滤无效数据
            if not isinstance(item, dict):
                logger.warning(f"OCR item is not a dict: {item}")
                continue
            if item.get("Score", 0.0) == 0.0:
                continue
            if item.get("Text", "") == "":
                continue
            ocr_data = OcrData(item)
            ocr_data.to_reference()  # 截图坐标 → 基准空间坐标
            logger.info(f"result: {ocr_data}")
            data_result.append(ocr_data)

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"OCR detection took {elapsed_ms:.2f} ms")
        return data_result


class RuleOcr:
    """文字识别

    用法1：
    ```python
    ruleocr = RuleOcr(asset)
    if result := ruleocr.match():
        Mouse.click(result.center, *args, **kwargs)
    ```

    用法2：
    ```python
    result = RuleOcr().get_raw_result()
    for item in result:
        if target_text == item.text:
            do something
        if target_text in item.text:
            do something

    ```
    """

    def __init__(
        self,
        assetocr: AssetOcr = None,
        name: str = None,
        keyword: str = None,
        region: tuple = None,  # 暂未用上
        score: float = 0.7,
        method: Literal["PERFECT", "INCLUDE"] = "PERFECT",
    ) -> None:
        """
        Args:
            assetocr (AssetOcr):  文字识别资源
            name (str): 名称
            keyword (str): 关键词
            region (tuple): 区域
            score (float): 识别阈值
            method (Literal["PERFECT", "INCLUDE"]): 匹配方式，PERFECT：完全匹配，INCLUDE：包含匹配
        """
        if assetocr:
            self.keyword = assetocr.keyword
            self.name = assetocr.name
            self.region = assetocr.region
            self.score = assetocr.score
            self.method = assetocr.method
        else:
            self.keyword = keyword
            self.name = name
            self.region = region
            self.score = score
            self.method = method

        if self.region is None or self.region == (0, 0, 0, 0):
            self.region = window_manager.current.client_rect
        else:
            # 素材/显式 region 是基准空间坐标 → 实际客户区
            self.region = scale_region(self.region)

        self.match_result: OcrData = None
        self.detector = OcrDetector(self.region)

    def get_raw_result(self) -> list[OcrData]:
        """获取原始OCR检测结果

        Returns:
            list[OcrData]: OCR检测结果列表
        """
        return self.detector.get_raw_result()

    def match(
        self,
        ocr_result: list[OcrData] = None,
        keyword: str = None,
        score: float = None,
        debug: bool = False,
    ) -> OcrData | None:
        """执行文字匹配

        Args:
            ocr_result: 预计算的OCR结果，如果为None则自动计算
            keyword: 要匹配的关键词，如果为None则使用初始化时的关键词
            score: 匹配阈值，如果为None则使用初始化时的阈值
            debug: 是否开启调试模式

        Returns:
            OcrData | None: 匹配结果
        """
        if not ocr_manager.is_initialized():
            ocr_manager.init()

        # 重置匹配结果
        self.match_result = None

        if ocr_result is None:
            ocr_result = self.get_raw_result()

        if keyword is None:
            keyword = self.keyword
        if score is None:
            score = self.score

        for item in ocr_result:
            if item.score < score:
                continue
            if self.method == "PERFECT":
                if item.text == keyword:
                    self.match_result = item
                    return item
            elif self.method == "INCLUDE":
                if keyword in item.text:
                    self.match_result = item
                    return item

        return None


def ocr_match_once(asset_list: list[AssetOcr]) -> RuleOcr | None:
    """批量文字匹配（一次截图，多次匹配）

    对同一次截图结果执行多个关键词匹配，避免重复截图开销。
    返回第一个匹配成功的 RuleOcr 实例，调用者通过 .match_result 获取匹配结果。

    Args:
        asset_list (list[AssetOcr]): AssetOcr 列表，每个元素定义一组匹配规则（关键词、阈值、匹配方式）

    Returns:
        RuleOcr | None: 第一个匹配成功的 RuleOcr 实例，全部未匹配返回 None
    """
    # 使用OcrDetector获取一次OCR结果，避免重复截图
    detector = OcrDetector()
    ocr_result = detector.get_raw_result()

    for item in asset_list:
        rule = RuleOcr(item)
        result = rule.match(ocr_result)
        if result:
            return rule

    return None
