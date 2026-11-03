"""入库编排层。

不要在这里 re-export ingest，否则 `python -m ragkb.pipeline.ingest`
会触发 runpy 的双重导入警告。
"""
