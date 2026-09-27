from __future__ import annotations

# =============================================================================
# 02_tools_custom.py —— 自定义工具：从 type hints 自动生成 JSON Schema
# =============================================================================
#
# 这是框架里"最像魔法"的一层，也是面试最好讲的一层。
#
# 传统写法是手搓一份 JSON Schema：
#
#     {"name": "search", "parameters": {"type": "object",
#      "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}
#
# liteagent 的写法是**写一个普通的 Python 函数**：
#
#     @tool
#     def search(keyword: str, limit: int = 5) -> str:
#         """Search the catalogue.
#
#         Args:
#             keyword: what to search for
#             limit: max number of hits
#         """
#
# 装饰器会在**定义时**（不是调用时）反射出三样东西：
#   1. 类型注解   -> parameters.properties 里的 type / items / enum / 嵌套 object
#   2. docstring  -> 函数级 description，以及 `Args:` 段里每个参数的 description
#   3. 默认值     -> 是否进 required（无默认值且非 Optional 才是必填）
#
# 本文件按这个顺序演：
#   §1 最小例子                —— 一个函数 -> 一份 schema
#   §2 参数形态全景            —— int/str/bool、Optional、Literal、list[str]、嵌套 dataclass
#   §3 把 schema 打出来看      —— json.dumps(indent=2)，这是本文件的主角
#   §4 模型实际看到的东西      —— ToolRegistry.schemas("openai"/"anthropic") 与 to_prompt()
#   §5 入参校验                —— 模型传错参数时，在**你的函数被调用之前**就被拦住
#   §6 端到端                  —— 让 Agent 真的调用这个自定义工具（离线）
#
# 运行方式（默认离线，不联网、不需要 key）：
#     python3 examples/02_tools_custom.py --offline
#     python3 examples/02_tools_custom.py            # 不写 --offline 也是离线

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

# ---- 让 examples/ 下的脚本"直接 python3 就能跑"（详见 01_quickstart.py 的说明）----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from liteagent import (  # noqa: E402 - 必须在 sys.path 引导之后
    Agent,
    AgentConfig,
    ScriptedLLM,
    ScriptedResponse,
    ToolRegistry,
    get_llm,
    tool,
)
from liteagent.tools.schema import Param, validate_instance  # noqa: E402

LINE = "-" * 74
BAR = "=" * 74


def section(title: str) -> None:
    """统一的分节标题 —— 示例的 stdout 是要给人读的，结构比信息量重要。"""
    print()
    print(BAR)
    print(title)
    print(BAR)


# =============================================================================
# §1 最小例子：一个普通函数 + 一个装饰器
# =============================================================================
# 注意 docstring 的格式：首行是函数的 description（会进 prompt，越短越好），
# 空行之后的 `Args:` 段是逐参说明。框架同时认 Google 风格（`Args:`）与
# Sphinx 风格（`:param x:`），也容忍完全没有 docstring —— 只是那时 description 为空串，
# 模型就得靠参数名猜意图，效果通常很差。


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


# =============================================================================
# §2 参数形态全景
# =============================================================================
# 下面的工具故意把"模型可能传的各种参数形态"凑齐，每一种都对应 schema 里的一处规则。
# 逐条对照 §3 打印出来的 schema 阅读效率最高。


@dataclass
class GeoPoint:
    """A geographic coordinate (nested dataclass -> nested JSON object)."""

    lat: float
    lon: float


@tool
def search_products(
    keyword: str,
    limit: int = 5,
    include_out_of_stock: bool = False,
    category: Optional[str] = None,
    sort_by: Literal["price", "rating", "newest"] = "price",
    tags: Optional[list[str]] = None,
    near: Optional[GeoPoint] = None,
    min_price: Annotated[float, Param(description="最低价（含税），人民币元", ge=0)] = 0.0,
) -> str:
    """Search the product catalogue by keyword.

    Args:
        keyword: What to search for (the only required field).
        limit: Maximum number of hits to return.
        include_out_of_stock: Whether sold-out items are included.
        category: Optional category filter.
        sort_by: Sort order of the returned hits.
        tags: Optional tag filters; a product must carry every tag.
        near: Optional coordinate; hits are ordered by distance when given.
        min_price: Lower bound (inclusive) on the price filter.
    """
    parts = [f"keyword={keyword!r}", f"limit={limit}", f"sort_by={sort_by!r}"]
    if category is not None:
        parts.append(f"category={category!r}")
    if include_out_of_stock:
        parts.append("include_out_of_stock=True")
    if tags:
        parts.append(f"tags={tags!r}")
    if near is not None:
        parts.append(f"near=({near.lat}, {near.lon})")
    if min_price:
        parts.append(f"min_price={min_price}")
    return "search_products(" + ", ".join(parts) + ")"


#: 本文件演示用的工具集合。**不要**用全局默认注册表：
#: 显式 `ToolRegistry([...])` 才能保证"这个示例注册了哪些工具"是文件里一眼可见的。
REGISTRY = ToolRegistry([add, search_products])


# =============================================================================
# §3 把生成的 schema 打出来
# =============================================================================
def demo_schema() -> None:
    """打印 add 与 search_products 的 OpenAI 格式 schema（缩进 JSON）。"""
    section("§1 + §3  @tool 生成的 JSON Schema（这就是模型看到的工具说明书）")

    print("输入 —— 一个普通函数（本文件 §1 里的 add，一行 schema 都没手写）：")
    print("    @tool")
    print("    def add(a: int, b: int) -> int:")
    print('        """Add two integers and return the sum.')
    print("")
    print("        Args:")
    print("            a: The first addend.")
    print("            b: The second addend.")
    print('        """')
    print("        return a + b")
    print()
    print("输出 —— 反射出来的 schema（json.dumps(add.to_openai_schema(), indent=2)）：")
    print(json.dumps(add.to_openai_schema(), ensure_ascii=False, indent=2))

    print()
    print(LINE)
    print("参数形态全景 —— search_products 的 schema：")
    print(LINE)
    print(json.dumps(search_products.to_openai_schema(), ensure_ascii=False, indent=2))

    # 逐条对照表：把 §7.1 的映射规则摊开写，比读 schema 更好记。
    print()
    print("对照表（Python 注解 -> JSON Schema）：")
    rows = [
        ("keyword: str", "type=string，**无默认值 -> 进 required**"),
        ("limit: int = 5", "type=integer，有默认值 -> 不进 required"),
        ("include_out_of_stock: bool = False", "type=boolean（bool 必须先于 int 判定，"
                                               "否则 True 会被当成 integer）"),
        ("category: Optional[str] = None", "type=string；Optional 只表达"
                                           '"非必填"，**不生成** type=[string,null]（D-05）'),
        ('sort_by: Literal["price", ...]', "type=string + enum=[...]；"
                                           "枚举值同类型时才带 type，混型则省略"),
        ("tags: Optional[list[str]]", "type=array + items={type:string}"),
        ("near: Optional[GeoPoint]", "嵌套 dataclass -> 内联成 nested object"
                                     "（不是 $ref，provider 兼容性更好）"),
        ("Annotated[float, Param(ge=0)]", "数值约束 -> minimum=0；"
                                          "Param 是零依赖的元数据载体，不引入 pydantic"),
    ]
    width = max(len(left) for left, _ in rows)
    for left, right in rows:
        print(f"  {left:<{width}}  ->  {right}")

    print()
    print("两个值得单独讲的细节：")
    print("  * required 只有 ['keyword']：公式是"
          "「没有默认值 且 不是 Optional 且 Param 没给 default」。")
    print("    所以 `category: Optional[str] = None` 与 `category: Optional[str]` 都不进 required。")
    print("  * 每个 object 都带 additionalProperties=false："
          "模型编造一个不存在的参数名时，可在调用前被拒掉，")
    print("    而不是悄悄传进你的函数、在函数体里变成一条诡异的 TypeError。")
    print()
    print("还有一个容易被忽略的坑：本项目每个 .py 的第一行都是")
    print("    from __future__ import annotations")
    print("（PEP 563），于是 `inspect.signature` 拿到的是字符串 'int' 而不是类型对象。")
    print("所以 schema 反射**必须先解析字符串注解**再判断，否则上面整张表会在 3.10 上全体失效。")
    print("证据：同一份 schema 生成正确，而 Tool.to_dict()['signature'] 打印出来是")
    print(f"    {add.to_dict()['signature']!r}  <- 带引号，正是字符串注解的形态。")


# =============================================================================
# §4 模型实际看到的东西
# =============================================================================
def demo_registry_schemas() -> None:
    """同一个 ToolRegistry，导出成不同 provider / 不同用途的形态。"""
    section("§4  同一个工具集，四种导出形态")

    print("[a] 原生 function-calling：ToolRegistry.schemas(fmt=\"openai\")")
    print("    这一份会被原样塞进 chat/completions 的 `tools=` 参数。")
    openai_payload = REGISTRY.schemas(fmt="openai")
    print(f"    共 {len(openai_payload)} 条；摘录 name 字段："
          f"{[item['function']['name'] for item in openai_payload]}")
    print()
    print("    add 那一条的完整原文：")
    print(json.dumps(openai_payload[0], ensure_ascii=False, indent=2))

    print()
    print("[b] Anthropic Messages API：ToolRegistry.schemas(fmt=\"anthropic\")")
    print("    同样的信息，字段名不同（input_schema 而不是 parameters）。")
    print(json.dumps(REGISTRY.schemas(fmt="anthropic")[0], ensure_ascii=False, indent=2))

    print()
    print("[c] 文本 ReAct 模式：ToolRegistry.to_prompt(fmt=\"text\")")
    print("    文本模式没有 tools= 参数，工具清单要**渲染进 system prompt**，")
    print("    所以这里是紧凑的一行一个（每行都会逐字进 token 预算）。")
    print(REGISTRY.to_prompt(fmt="text"))

    print()
    print("[d] 调试/落盘用：Tool.to_dict()")
    print("    多带 signature / is_async / dangerous / idempotent 等运行期属性，")
    print("    排查「模型为什么传错参数」时最先看的就是 signature。")
    debug_view = add.to_dict()
    print(f"    add.to_dict() keys = {sorted(debug_view)}")
    print(f"    signature = {debug_view['signature']!r}")


# =============================================================================
# §5 入参校验：在你的函数被调用之前
# =============================================================================
def demo_validation() -> None:
    """ToolExecutor 在调用工具前会跑一遍 schema 校验；这里直接调用那个纯函数。

    为什么要单独讲：`@tool` 生成的 schema 不只是"给模型看的说明书"，
    它是**双向契约** —— 同一份 schema 既约束模型的输出，也用来校验模型的入参。
    一份 schema 两处用，就不存在"提示词说 A、校验器认 B"的漂移。

    注意：`Tool.run()` / `Tool.arun()` **刻意不做校验**（校验只在 executor 里做一次），
    所以这里演示的是 executor 内部用的那个函数 `validate_instance(instance, schema)`，
    它返回**错误消息列表**（空列表 = 通过），不是抛异常。
    """
    section("§5  入参校验：同一份 schema，双向契约")
    schema = search_products.parameters  # 即 schema["function"]["parameters"]

    cases: list[tuple[str, dict[str, Any]]] = [
        ("合法调用", {"keyword": "keyboard"}),
        ("类型错了（limit 传字符串）", {"keyword": "keyboard", "limit": "many"}),
        ("缺必填（没给 keyword）", {"limit": 3}),
        ("枚举值非法（sort_by）", {"keyword": "keyboard", "sort_by": "cheapest"}),
        ("约束越界（min_price < 0）", {"keyword": "keyboard", "min_price": -1}),
        ("多给了一个不存在的参数", {"keyword": "keyboard", "colour": "red"}),
    ]
    for label, payload in cases:
        errors = validate_instance(payload, schema)
        if not errors:
            print(f"  [OK ] {label}")
            # 校验通过 -> 才轮到你的函数真正执行。这里直接调用原始函数演示效果。
            print(f"        -> {search_products.run(payload)}")
        else:
            print(f"  [ERR] {label}")
            for error in errors:
                print(f"        {error}")

    print()
    print("  （真实链路里，这些错误消息不会直接抛给用户：ToolExecutor 会把它包成")
    print("    一个 ok=False 的 ToolResult 回灌给模型，让模型自己修参数后再试一轮。）")


# =============================================================================
# §6 端到端：让 Agent 真的调用这个自定义工具
# =============================================================================
async def demo_agent(args: argparse.Namespace) -> int:
    """把 §2 的工具交给 Agent，跑一轮完整的 ReAct 循环。

    离线剧本（ScriptedLLM）：
        step 1: 模型要求调用 search_products(keyword="mechanical keyboard", limit=2,
                                              sort_by="rating", tags=["wireless"])
        step 2: 模型给出最终答案
    """
    section("§6  端到端：Agent 调用自定义工具（离线，ScriptedLLM 驱动）")

    if args.provider is not None:
        print(f"注意：--provider {args.provider!r} 走真实模型；"
              "下面这段**演示脚本假定**的是离线剧本，联网时输出会不同。")
        try:
            llm: Any = get_llm(args.provider if args.model is None
                               else f"{args.provider}:{args.model}")
        except Exception as exc:
            print(f"装配失败：{type(exc).__name__}: {exc}")
            return 2
    else:
        llm = ScriptedLLM(
            [
                # 一个"参数传得比较满"的工具调用：正好覆盖 §2 里的 list[str] 与 Literal。
                ScriptedResponse.tool(
                    "search_products",
                    {
                        "keyword": "mechanical keyboard",
                        "limit": 2,
                        "sort_by": "rating",
                        "tags": ["wireless"],
                        "include_out_of_stock": False,
                    },
                ),
                ScriptedResponse.text(
                    "Found 2 wireless mechanical keyboards, sorted by rating."
                ),
            ],
            model="scripted-1",
        )

    agent = Agent(
        llm=llm,
        tools=REGISTRY,
        # mode="auto"（默认）会问 llm.resolve_mode(has_tools=True)：
        # ScriptedLLM.supports_tool_calling 为 True，于是自动选原生 function-calling。
        config=AgentConfig(max_steps=5),
        name="tools-demo",
    )
    print(f"agent.describe() = {agent.describe()}")

    result = await agent.arun("Find me a top-rated wireless mechanical keyboard.")
    print()
    print(f"status = {result.status.value}   steps = {result.steps}   "
          f"usage.total_tokens = {result.usage.total_tokens}")
    print(f"output = {result.output!r}")
    for call, tool_result in zip(result.tool_calls, result.tool_results):
        print(f"  模型说要调用 : {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        print(f"  实际执行结果 : {tool_result.content!r}   ok={tool_result.ok}")
    return 0 if result.ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="liteagent 示例 02：自定义工具与自动 JSON Schema 生成（默认离线）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线运行（默认；用 ScriptedLLM 驱动 §6 的 Agent，不联网、不需要 API key）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="§6 改用真实 provider（如 openai）；§1-§5 的 schema 演示与此无关",
    )
    parser.add_argument("--model", default=None, help="配合 --provider 指定模型名")
    args = parser.parse_args(argv)
    if args.provider is None:
        args.offline = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    print(BAR)
    print("liteagent 示例 02 —— 自定义工具：@tool 如何从 type hints 自动生成 JSON Schema")
    print(f"运行模式：{'离线（--offline）' if args.offline else '联网 ' + str(args.provider)}")
    print(BAR)

    # §1-§5 是纯计算（反射 + 校验），不碰网络，任何模式下都跑。
    demo_schema()
    demo_registry_schemas()
    demo_validation()
    return asyncio.run(demo_agent(args))


if __name__ == "__main__":
    raise SystemExit(main())
