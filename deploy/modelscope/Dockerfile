# 魔搭 Docker 创空间镜像（服务必须监听 0.0.0.0:7860，端口不可改）。
#
# 本地验证：docker build -f deploy/modelscope/Dockerfile -t rag-kb:local .
# CI 验证：  GitHub Actions 会把 BASE_IMAGE 换成 python:3.11-slim，
#            因为魔搭的镜像仓库在境内，境外 runner 拉不动。
ARG BASE_IMAGE=modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/python:3.10
FROM ${BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 依赖单独一层：pymupdf 一个包就 60 MB，和业务代码分开装，
# 改代码时不用重装。COPY 顺序按"变动频率从低到高"排。
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"

# 代码、前端、语料、预建索引（索引里已有向量，启动只做加载，不调 embedding）
COPY web ./web
COPY scripts ./scripts
COPY corpus_full ./corpus_full
COPY eval ./eval
COPY deploy/qdrant_seed ./deploy/qdrant_seed

# 默认走硅基流动。密钥不写进镜像，由创空间的 Secret 在运行时注入。
ENV KB_HOST=0.0.0.0 \
    KB_PORT=7860 \
    KB_DOCS=corpus_full \
    KB_EMBEDDER=api \
    KB_EMBED_BASE_URL=https://api.siliconflow.cn/v1 \
    KB_EMBED_MODEL=Qwen/Qwen3-Embedding-8B \
    KB_RERANKER=siliconflow \
    SILICONFLOW_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
    KB_LLM=siliconflow \
    KB_MODEL=Qwen/Qwen3-8B \
    KB_DISABLE_THINKING=1 \
    KB_EXAMPLES_FILE=eval/deploy_examples.json

EXPOSE 7860

# 索引和阈值成对出现：换了 embedding 就必须重新标定，所以这里显式指定阈值文件，
# 不让它按目录名去猜（猜错的表现是"该答的全拒"，日志里一句报错都没有）。
CMD ["python", "scripts/serve.py", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--docs", "corpus_full", \
     "--seed", "deploy/qdrant_seed", \
     "--threshold-file", "eval/thresholds_siliconflow.json"]
