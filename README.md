# auto-dual-python-package
An automated Lagrangian dual derivation engine for Operations Research and convex optimization models, powered by SymPy. It translates Python-based multi-index algebraic models into their analytical dual forms and generates academic-ready LaTeX code.
# DualDeriver: 运筹学自动对偶推导工具使用手册

`DualDeriver` 是一个基于 Python `SymPy` 的符号计算引擎。它允许你通过编写极其贴近纯数学公式的 Python 代码，自动推导线性规划（LP）或凸优化问题的拉格朗日对偶模型，并直接输出可用于论文的 LaTeX 代码。

---

## 核心建模流程 (6步法)

```python
import sympy as sp
from dual_deriver import DualDeriver, SUM, TERM, render_dual

# 1. 初始化推导器（明确原问题是求 min 还是 max）
deriver = DualDeriver(sense="min")

# 2. 定义纯数学符号（循环指标），如 i, j, k
i, j = sp.symbols("i j")

# 3. 定义已知参数（Parameters），如成本 c_ij，需求 d_i
# 不需要提前指定维度，直接声明为 IndexedBase 即可
c = sp.IndexedBase("c")
d = sp.IndexedBase("d")
U = sp.IndexedBase("U")

# 声明变量 y_{ij} >= 0
y = deriver.declare_var(
    name="y",                 # LaTeX 中显示的变量基础名
    index_sets=["I", "J"],    # 变量所在的集合名称（用于内部逻辑匹配）
    free_symbols=[i, j],      # 绑定的数学指标符号
    lower=0                   # 定义下界为0（如果是无约束自由变量，设为 None 或省略）
)

# 4. 设定目标函数: min \sum_{i \in I} \sum_{j \in J} c_{ij} y_{ij}
deriver.set_objective(
    SUM(c[i,j] * y[i,j], (i, "I"), (j, "J"))
)

# 5. 录入约束: \sum_{j \in J} y_{ij} >= d_i, \forall i \in I
deriver.add_constraint(
    lhs = SUM(y[i,j], (j, "J")),   # 左侧表达式 (LHS)
    sense = ">=",                  # 约束符号：">=", "<=", 或 "=="
    rhs = TERM(d[i]),              # 右侧表达式 (RHS)
    
    # 全称量词 (forall) 设定
    forall = [i],                  # 这条约束需要遍历的指标
    forall_sets = ["I"],           # 指标对应的集合名称
    
    multiplier_name = "lambda",    # 自动生成对应的对偶乘子（LaTeX输出为 \lambda_i）
    name = "demand_satisfaction"   # 约束别名（可选，方便调试）
)

# 6. 执行推导并输出
    result = deriver.derive_dual()
    print("====== Generated Dual LaTeX ======")
    print(render_dual(result))

