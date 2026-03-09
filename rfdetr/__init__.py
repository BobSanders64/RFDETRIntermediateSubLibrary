# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import sys

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if sys.platform == "win32":
    try:
        import triton  # noqa: F401
    except ImportError:
        print(
            "\n"
            "============================================================\n"
            " TRITON NOT FOUND (required for xformers on Windows)\n"
            " Please run the command in the README (or seen below):\n"
            " pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl \n"
            " if your on something other than Windows, you can install it normally via pip3 install triton"
            "\n"
            "============================================================\n"
        )
else:
    try:
        import triton
    except ImportError:
        print("""Please install triton by running:\n
        pip3 install triton\n
        """)

from rfdetr.detr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRLargeDeprecated,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegIntermediate,
    RFDETRSegNano,
    RFDETRSegPreview,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)
from rfdetr.platform.models import (
    RFDETR2XLarge,
    RFDETRXLarge,
)
