# PyInstaller spec for Pikachu Volleyball desktop app
# Build with: .venv/bin/pyinstaller pikachu_volleyball.spec
#
# The app bundles: game code, ONNX model, pikazoo env + assets, onnxruntime

import os
import sys

pikazoo_pkg = os.path.join(
    os.path.dirname(sys.executable),
    '..', 'lib', 'python3.12', 'site-packages', 'pikazoo'
)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/pikachu_actor.onnx', 'models'),
        ('assets/img/*.png', 'pikazoo/env/img'),
    ],
    hiddenimports=[
        'pikazoo',
        'pikazoo.env',
        'pikazoo.env.pikazoo_env',
        'pikazoo.env.physics',
        'pikazoo.env.cloud_and_wave',
        'onnxruntime',
        'pygame',
        'numpy',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['torch', 'matplotlib', 'tensorboard', 'tqdm'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PikachuVolleyball',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
