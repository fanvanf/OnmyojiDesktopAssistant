import hashlib
from pathlib import Path

import httpx

from .application import MODEL_DIR_PATH, Connect
from .log import logger

OCR_VERSION: str = "PP-OCRv6"
"""OCR 模型版本"""

MODEL_TYPE: str = "small"
"""模型规格。

注意：可用取值随 `OCR_VERSION` 变化，v4/v5 为 `mobile`/`server`，v6 为 `tiny`/`small`/`medium`。
"""

LANG_TYPE: str = "ch"
"""识别语言，中英文"""

ENGINE_TYPE: str = "onnxruntime"
"""推理引擎"""


class RapidModel:
    """RapidOCR 模型

    每个模型是一个独立的 `.onnx` 文件，下载地址与校验和从 RapidOCR 自带的
    `default_models.yaml` 中解析，避免与库版本脱节。
    """

    task_type: str = ""
    """模型任务类型，det 检测 / rec 识别"""

    @classmethod
    def get_model_info(cls) -> dict:
        """从 RapidOCR 获取该模型的下载信息

        Returns:
            dict: 含 `model_dir`(下载地址) 与 `SHA256`(校验和)
        """
        from rapidocr.inference_engine.base import FileInfo, InferSession
        from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

        return InferSession.get_model_url(
            FileInfo(
                engine_type=EngineType(ENGINE_TYPE),
                ocr_version=OCRVersion(OCR_VERSION),
                task_type=TaskType(cls.task_type),
                lang_type=LANG_TYPE,
                model_type=ModelType(MODEL_TYPE),
            )
        )

    @classmethod
    def get_model_path(cls) -> Path:
        """模型在 `models` 目录下的存放路径

        Returns:
            Path: 模型文件路径
        """
        info = cls.get_model_info()
        return MODEL_DIR_PATH / Path(info["model_dir"]).name

    @classmethod
    def is_valid(cls) -> bool:
        """校验模型文件是否存在且完整"""
        model_path = cls.get_model_path()
        if not model_path.is_file():
            logger.ui_error(f"{model_path.name} 不存在")
            return False

        expect_sha256 = cls.get_model_info().get("SHA256")
        if expect_sha256 is None:
            return True

        actual_sha256 = cls.get_file_sha256(model_path)
        if actual_sha256 != expect_sha256:
            logger.ui_error(f"{model_path.name} 校验失败，文件可能已损坏")
            return False

        return True

    @classmethod
    def download(cls) -> bool:
        """下载模型到 `models` 目录

        Returns:
            bool: 是否下载成功
        """
        info = cls.get_model_info()
        model_path = cls.get_model_path()
        model_name = model_path.name

        logger.ui(f"下载 {model_name} 模型...")
        if not cls._download_file(info["model_dir"], str(model_path)):
            return False

        if not cls.is_valid():
            model_path.unlink(missing_ok=True)
            return False

        return True

    HTTP_TIMEOUT: int = 60
    """HTTP 请求超时时间（秒）"""

    CHUNK_SIZE: int = 8192
    """下载文件时的块大小（字节）"""

    PROGRESS_STEP: int = 10
    """进度日志输出间隔（百分比）"""

    @classmethod
    def _download_file(cls, url: str, save_path: str) -> bool:
        """下载文件（带进度条）"""
        try:
            with httpx.stream(
                "GET", url, headers=Connect.headers, timeout=cls.HTTP_TIMEOUT, follow_redirects=True
            ) as resp:
                resp.raise_for_status()

                total_size = int(resp.headers.get("content-length", 0))
                downloaded_size = 0
                last_logged_percent = 0

                with open(save_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=cls.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            # 每 PROGRESS_STEP% 输出一次进度
                            if total_size > 0:
                                progress = downloaded_size / total_size * 100
                                current_percent = int(progress // cls.PROGRESS_STEP * cls.PROGRESS_STEP)
                                if current_percent != last_logged_percent and current_percent % cls.PROGRESS_STEP == 0:
                                    last_logged_percent = current_percent
                                    logger.ui(
                                        f"下载进度: {progress:.1f}% ({downloaded_size / 1024 / 1024:.2f}MB/{total_size / 1024 / 1024:.2f}MB)"
                                    )
            return True
        except httpx.HTTPError as e:
            logger.ui_error(f"下载失败: {e}")
            Path(save_path).unlink(missing_ok=True)
            return False
        except OSError as e:
            logger.ui_error(f"文件操作失败: {e}")
            return False
        except Exception as e:
            logger.ui_error(f"下载过程中发生未知错误: {e}")
            Path(save_path).unlink(missing_ok=True)
            return False

    @classmethod
    def get_file_sha256(cls, file_path: Path) -> str:
        """计算文件的 sha256"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(cls.CHUNK_SIZE), b""):
                sha256.update(block)
        return sha256.hexdigest()


class DetModel(RapidModel):
    """文本检测模型"""

    task_type: str = "det"


class RecModel(RapidModel):
    """文本识别模型"""

    task_type: str = "rec"


ALL_MODELS: tuple = (DetModel, RecModel)
"""程序运行所需的全部模型"""


def check_models() -> bool:
    """检查模型文件是否完整

    Returns:
        bool: 是否完整
    """
    if not MODEL_DIR_PATH.exists():
        logger.error("models文件夹不存在")
        return False

    for model in ALL_MODELS:
        if not model.is_valid():
            return False

    logger.info(f"{OCR_VERSION} 的模型文件完整")
    return True


def download_models() -> bool:
    """下载全部模型

    Returns:
        bool: 是否全部下载成功
    """
    for model in ALL_MODELS:
        if not model.download():
            return False

    logger.ui("模型下载完成")
    return True


def auto_download() -> bool:
    """检查并按需下载模型

    Returns:
        bool: 模型是否可用
    """
    logger.info(f"检查 {OCR_VERSION} 的模型文件...")

    if check_models():
        logger.info("模型已存在且完整，无需下载")
        return True

    logger.ui(f"模型不完整或不存在，开始下载 {OCR_VERSION} 的模型...")
    if not download_models():
        logger.ui_error(f"下载 {OCR_VERSION} 模型失败")
        return False

    return check_models()
