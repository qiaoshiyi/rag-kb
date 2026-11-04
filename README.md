# rag-kb

一个自己搭的企业知识库 RAG 问答系统：txt / md / pdf / docx 四类文档、多轮追问、带引用作答、可量化评测。

没有用 LangChain / LlamaIndex，解析、切分、检索、改写、生成、评测都是自己写的。

![控制台](https://cdn.jsdelivr.net/gh/qiaoshiyi/rag-kb@main/assets/screenshot.png)

<sub>两轮追问：第二句「另一个档次呢？」被改写成完整问句后重新检索。截图跑的是早期 16 篇语料，所以顶部显示 110 个切片；换成 `corpus_full` 是 494 块。</sub>

## 为什么做这个

我想把 RAG 的每一层都自己写一遍，看看到底是哪一层在决定效果。用框架能更快跑起来，  
但出问题时只能读框架源码——而这个项目里一半的问题恰恰出在解析和评测口径上，正是框架最不透明的两块。

所以它是一个**学习型项目**，不是生产系统：语料是造的，部署是单机内存版，没有权限体系。  
但每一层的取舍、每一处失败、每一个数字都能追到我自己的判断，这一点是它存在的理由。

## 现在能跑到什么程度

12 份制度、2.6 万字、494 个切片，真实 API（`qwen3.7-text-embedding-flash` + `qwen3.7-text-rerank` + `deepseek-v4.1-flash`）：

| 项             | 结果                                            |
| --- | --- |
| 检索（39 条常规题）   | hit@5 **1.000**，MRR 0.974                     |
| 检索（19 条语义改述题） | hit@5 **1.000**，MRR 0.947                     |
| 拒答阈值          | **0.748**（正样本下限 0.926 / 负样本上限 0.571，间隔 0.355） |
| 多轮（12 条零内容追问） | span MRR **0.958**（裸问 0.736）                  |
| 回答质量（54 条）    | 拒答准确率 **1.000**、引用对位 **1.000**、要点覆盖 0.962     |
| 测试            | **142 个**，全部离线可跑                              |

这些数字怎么来的、以及它们怎么骗过我，写在 [docs/踩坑复盘.md](docs/踩坑复盘.md)。

## 跑起来

**环境要求：只要有 Python 3.10+**（3.11 / 3.12 / 3.13 都验证过）。  
不需要 GPU、不需要 Docker、不需要单独装 Qdrant（默认跑内存模式），占磁盘约 300 MB。

```bash
git clone https://github.com/qiaoshiyi/rag-kb.git && cd rag-kb
pip install -e ".[dev,api]"      # 约 40 秒，会装 60 个包
```

下载 zip 也行，解压后进目录一样执行。`[dev]` 是测试用的，`[api]` 是服务层用的；少装 `[api]` 就起不了 HTTP 服务，少装 `[dev]` 就跑不了测试。

不需要 API key 也能先看见东西在动（用确定性替身：hashing 向量 + 词面重排 + 抽取式作答）：

```bash
python -m pytest                    # 142 个测试，不联网，约 7 秒
python scripts/serve.py --offline   # 打开 http://127.0.0.1:8000
```

> **首次启动会停 12 秒左右，那不是卡住。** 它在读 12 份 txt/md/pdf/docx、切成 494 个父子块、
> 逐块算向量、写进 Qdrant 内存库。之后每个请求都是毫秒级。终端会打印 `索引 corpus_full · 12 篇 · 494 块` 确认。

接真实模型（OpenAI 兼容接口，这里用阿里云百炼的兼容网关）：

```bash
export DASHSCOPE_API_KEY=sk-xxx     # Windows 用: set DASHSCOPE_API_KEY=sk-xxx
python scripts/serve.py             # 默认索引 corpus_full，阈值自动配对
python scripts/ask.py "年假天数按工龄怎么分档？"
python scripts/chat_demo.py         # 多轮：看每轮改写后的检索问法
```

Qdrant 默认用 `:memory:`，不需要 Docker；接真实服务把 `QdrantClient(":memory:")` 换成 `QdrantClient(url=...)` 就行。

> 不装 `.[embed]` 时全部回退到 `HashingEmbedder` / `LexicalReranker`。  
> 它们只按字面重叠打分，**能证明链路通，不能代表检索质量**——这句话很重要，别拿替身的分数当结论。

### 跑不起来时，先对照这四条

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `No module named 'ragkb'` | 没执行 `pip install -e .`，或当前不在仓库根目录 | 回到仓库根目录重装一次 |
| 页面能开，但问什么都「拒答」 | 语料换了、阈值没跟着换 | 用 `scripts/serve.py` 启动（它把两者绑在一起）；手配 `KB_DOCS` 时必须同时设 `KB_MIN_SCORE` |
| `No module named 'uvicorn'` | 只装了 `pip install -e .` | 换成 `pip install -e ".[dev,api]"` |
| 8000 端口被占用 | 本机已有服务在跑 | 加 `--port 8010` |

### 换成你自己的文档

把文件丢进一个新目录（支持 `.txt / .md / .pdf / .docx`），先关掉阈值把召回跑通：

```bash
python scripts/serve.py --docs 你的目录 --threshold 0    # 0 = 不拒答，先看能不能搜到东西
```

确认检索正常后**必须重新标定阈值**。原始阈值是按 12 份制度文档的分数分布定的，换语料直接套会大面积误杀。标定要三样东西：

```bash
python scripts/calibrate.py \
  --docs 你的目录 \
  --positive 你的正样本.jsonl \
  --negative 你的负样本.jsonl \
  --out eval/thresholds_你的目录.json
```

- **正样本**：确实能在这个语料里答出来的问题，20 条就够（格式见 `eval/full_golden.jsonl`，每行一个 JSON，字段 `question`）
- **负样本**：这个语料里**根本没有**的问题，15 条左右（格式见 `eval/full_negative.jsonl`）
- 这一步会调 embedding 和 rerank 接口，需要 `DASHSCOPE_API_KEY`

> ⚠️ `--positive` 不传的话，脚本默认用 `eval/golden_set.jsonl`（那是**另一套语料**的金标）。
> 拿它标你的语料，采到的是错配分数，算出来的阈值比不设还危险。

标定完正常启动即可（`serve.py` 按目录名找 `eval/thresholds_<目录名>.json`，找不到就不启用阈值）。

## 链路

```
四类文档 → 统一 IR(Block/Document) → 父子块切分 → 双向量写入 Qdrant
                                                      (dense + sparse)

query → ┬ 改写（多轮）──→ dense top30 ┐
        └ 原问题兜底 ───→ sparse top30 ┴→ RRF → 重排 20→6 → 阈值拒答 → 父块回填 → 带引用作答
```

## 技术栈

| 层 | 选型 | 为什么是它 |
| --- | --- | --- |
| 语言 / 打包 | Python 3.10+，`pyproject.toml` + setuptools | — |
| 解析 | PyMuPDF（pdf 文字层）、MinerU（扫描件 OCR，可选）、mammoth（docx→HTML）、markdown-it-py（md）、charset-normalizer（txt 编码探测） | 每类文档用它最擅长的工具，最后都收敛到同一个 IR |
| 切分 | 自研层级切分（父子块） | 要按标题层级切、还要父子两套尺寸，现成 chunker 不支持 |
| 向量库 | Qdrant（`qdrant-client`，内存 / 远程） | 原生支持 dense + sparse 双向量，过滤条件能下推 |
| Embedding | `qwen3.7-text-embedding-flash`（API，1024 维）／ bge-m3（本地，dense+sparse）／ Hashing（替身） | 三档可切：没 key 能跑，有 GPU 能本地跑 |
| 稀疏向量 | 自研词法编码（term hash → tf 权重） | 这个 API 的 embedding 只返回稠密，混合检索的另一半得自己补；不依赖分词器和语料统计 |
| 重排 | `qwen3.7-text-rerank`（DashScope 原生端点）／ bge-reranker（本地）／ 词面重叠（替身） | cross-encoder 对精确问法收益最大：20 条候选挑 6 条 |
| 生成 | `deepseek-v4.1-flash`（OpenAI 兼容接口） | 便宜、长上下文；但它带 reasoning，所以要处理"想太久返回空串" |
| 服务 | FastAPI + uvicorn，SSE 流式 | 流式输出用 SSE 就够，不需要 WebSocket |
| 前端 | 原生 HTML/CSS/JS 单文件 | 无构建、无依赖、无 CDN，离线内网也能开 |
| 测试 | pytest，142 个，全部离线 | 替身实现让测试不需要 API key |

核心依赖：

```bash
qdrant-client>=1.9   openai>=1.30   pymupdf>=1.24
mammoth>=1.7        markdown-it-py>=3.0   charset-normalizer>=3.3
fastapi>=0.110      uvicorn[standard]>=0.29        # pip install -e ".[api]"
```

三档实现靠 `build_embedder` / `build_reranker` / `build_llm` 三个工厂切换，`--offline` 一键全走替身。  
**替身只按字面重叠打分——能证明链路通，不能代表检索质量。**

## 项目结构

```
src/ragkb/
├── parsing/      txt / markdown / pdf（PyMuPDF + OCR 双通道）/ docx（mammoth→HTML）→ 统一 IR
├── chunking/     父子块切分（子 256–512，父 1024–2048，overlap 按类型 10/15/20%）
├── embedding/    API 版 / bge-m3 版 / 确定性替身；稀疏向量
├── indexing/     Qdrant 写入：双向量、uuid5 幂等、按文档删除、过滤
├── retrieval/    双路召回 + RRF + 重排 + 阈值拒答 + 父块回填
├── multiturn/    LLM 改写、可信度 guard、话题跳转
├── generation/   prompt 组装、引用解析、OpenAI 兼容实现、替身
├── evaluation/   hit/MRR/span/keyword、阈值标定、回答质量判分
└── api/          FastAPI：/ /search /ask /ask/stream（SSE）

corpus_full/      12 份语料（txt / docx / pdf / md 各 3）+ manifest.json
eval/             金标集、负样本、多轮集、四份标定结果
scripts/          造语料、检索演示、多轮演示、评测、标定、起服务、预建索引
web/index.html    单文件控制台，无构建、无依赖、无 CDN
tests/            142 个测试
docs/             技术选型、踩坑复盘
deploy/modelscope/ 创空间镜像（预建索引 + 7860 端口，可选）
.github/workflows/ CI：跑测试，并真的构建镜像起容器自检
```

约 1.1 万行（核心 4800 / 测试 1640 / 脚本 4200）。

## 几个我想解释的决定

**1. 融合用 RRF，不用加权求和。** 稠密分是余弦（0~1），稀疏分是词权重内积（无上界），  
量纲不同，直接相加必然被其中一路主导。RRF 只看名次，天然免疫，也不用调权重。

**2. 父块也入库，但不参加检索竞争。** 父块文本长、覆盖全，和任何问题的相似度都不低，  
放进去就会挤掉精确的子块。所以写入时带上 `kind`，检索时 `must_not kind=parent` 排除，  
只在命中子块后回填上下文。

**3. 拒答阈值必须标定，而且不能跨语料复用。** 它是分数分布的分位点，跟着 embedding、reranker、  
语料三样走。0.3 这个"经验默认值"在这套语料上会漏放 80% 的负样本。没标定就干脆不设阈值——  
宁可让 prompt 兜底，也别用假阈值误杀。

**4. 多轮改成双路召回。** 改写一旦跑偏（把「海外」理解成「海外子公司」），单路检索会把整轮带偏。  
原问题那一路权重 0.7 当兜底。改写结果还要过一道 guard：与最近两轮提问的相似度低于 0.30 就丢弃。

**5. docx 走 mammoth → HTML 中间层，而不是逐段遍历。** 逐段遍历只能拿到「文字 + 样式名」，  
标题层级和表格全靠猜。转成 HTML 后结构变成标准语义标签，和 Markdown 走同一条下游路径，  
**四类文档最后只维护一套解析逻辑**。

## 语料

为了能测出解析差异，我没有把同一段文字转成四种格式，而是让每种格式按它自己的习惯排：  
txt 是空格对齐的表和全角缩进（老系统导出的样子，GBK 编码），docx 有封面页和 Word 标题样式，  
pdf 是 A4 分页 + 宋体正文 + 黑体标题 + 页码，md 用 frontmatter 和围栏代码块。

| 格式   | 文档                                        | 编号                                  |
| --- | --- | --- |
| txt  | 员工考勤与假勤管理制度 / 差旅费用报销实施细则 / 信息资产分级与使用规范    | HY-HR-003 / HY-FIN-007 / HY-IT-011  |
| docx | 采购与招标管理办法 / 薪酬结构与绩效考核规定 / 产品质量检验与不合格品处置规程 | HY-PUR-002 / HY-HR-009 / HY-QA-005  |
| pdf  | 数据分级分类与安全管理规定 / 生产安全事故综合应急预案 / 来料与出货检验标准  | HY-SEC-001 / HY-EHS-004 / HY-QA-012 |
| md   | 研发项目管理制度 / 客户服务与投诉处理规范 / 运维值班与变更管理规范      | HY-RD-006 / HY-CS-008 / HY-IT-014   |

12 份文件同属一家虚构公司，有编号、版本、生效日期、责任部门、密级和修订记录，  
内容上互相引用（差旅规则引用费用制度、数据分级被安全规定引用），这样跨文档检索才有东西可测。

![PDF 语料](https://cdn.jsdelivr.net/gh/qiaoshiyi/rag-kb@main/assets/corpus_pdf.png)

<sub>其中一份 pdf 语料的实际页面：宋体正文、黑体标题、页眉页脚、表格跨页，标题层级靠字号推断。</sub>

```bash
python scripts/make_corpus_full.py --index   # 重新生成并打印解析统计
```

`corpus/` 是早期造的 16 篇短样例（评测脚本的默认值还指向它），`demo/` 是 8 篇最小样例、由  
`scripts/make_demo.py` 生成、不进仓库。想做真正的效果评估，用 `corpus_full/`。

## 踩过的坑

挑了五个最典型的，完整十条在 [docs/踩坑复盘.md](docs/踩坑复盘.md)：

| 坑             | 一句话                                                       |
| --- | --- |
| `第X条` 没被识别成标题 | 一份制度 21 块 → 60 块，10 个标题 → 38 个；补正则还要配「标题不以句读收尾」的守卫        |
| 文档级 hit 指标饱和  | 8 篇语料上三套策略全是 1.000，补 span 指标后才看出差别（0.736 / 0.889 / 0.958） |
| 评测集自己写错了      | 参考答案写成条款摘要，coverage 假跌到 0.816；关键词带语序、数字写法不一致              |
| 改写 guard 误杀   | 只跟上一轮比相似度会误判零内容追问，改成跟最近两轮取最大值（0.25 → 0.80）                |
| 演示层撒谎         | 语料 / 阈值 / 示例问题三处不自洽，自己跑评测全绿，别人打开满屏拒答                      |

最后一个坑的教训是：**指标好看 ≠ 用户能用**。所以现在页面上同时显示索引来源、拒答分数和拒答原因。

## 已知局限

- **跨文档综合题弱。** 「同时受保密协议和采购流程约束的情形」这类需要拼接两三份文件的题，3 条里会拒答 1 条。
- **LLM 判分有噪声。** 同一批 54 条答案跑两轮，faithful 会在 0.973 / 1.000 之间波动（±1 条）。
- **docx 只做了结构校验，没做视觉验收。** 本机没有 Word / WPS / LibreOffice，所以样式还原度没验证过。
- **语料是合成的。** 结构按真实企业制度写，但内容是我编的，不能代表真实文档的脏乱程度。
- **没有权限体系。** `acl_tags` 只是把过滤条件下推到检索，真正的鉴权不在这里。

## 文档

- [技术选型](docs/技术选型.md)：逐层决策表——为什么是这个解析器、这个切分粒度、这个融合算法，以及被否掉的备选方案
- [踩坑复盘](docs/踩坑复盘.md)：十条坑，现象 / 根因 / 修法，包括三次「指标在骗我」

## License

MIT

