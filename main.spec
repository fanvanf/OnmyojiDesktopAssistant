# -*- mode: python ; coding: utf-8 -*-
import importlib.metadata
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = []
binaries = []

# RapidOCR 资源：模型清单、默认配置、字典等
# 排除随包分发的默认模型（PP-OCRv6 small，约 31MB）：运行时统一使用 models 目录下的
# 模型文件，由 Global.model_root_dir 指定，不会读取包内 models
datas += [
    entry
    for entry in collect_data_files('rapidocr')
    if not entry[0].lower().endswith('.onnx')
]

# ONNX Runtime 原生库
try:
    importlib.metadata.version("onnxruntime-gpu")
    binaries += collect_dynamic_libs('onnxruntime-gpu')
except Exception:
    binaries += collect_dynamic_libs('onnxruntime')

# GPU 版：收集 nvidia 动态库
try:
    importlib.metadata.version("onnxruntime-gpu")
    binaries += collect_dynamic_libs('nvidia')
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OnmyojiDesktopAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon=['src/ui/buzhihuo.jpg'],
    contents_directory='lib',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='output',
)