# 理论推导文档索引

本目录是整个项目的理论单一事实源（single source of truth），覆盖：

- 无力传感器广义动量观测器；
- 固定工具与慢变偏置补偿；
- 末端 wrench 重构；
- 双机器人抓取矩阵、载荷分配与内力分解；
- 对象阻抗、分布式末端阻抗及 Virtual Linkage 阅读笔记；
- 物体质量和质心辨识；
- CHIP 风格柔顺学习与解析阻抗基线。

## 文件

- [`model_derivation_notes.md`](model_derivation_notes.md)：中文主笔记。用于持续记录假设、公式推导、实现映射、实验结果和待办。
- [`equations.tex`](equations.tex)：可独立编译的纯 LaTeX 公式库。用于论文和 PPT 复制公式。
- [`equations.pdf`](equations.pdf)：统一公式手册，已包含对象阻抗与末端分布阻抗等最新内容（2026-09-03 更新，共 4 页）。带日期的 `equations_2026-09-03.pdf` 保留为备份。

## 推荐工作流

1. 新结论先写入 `model_derivation_notes.md`，注明日期、适用边界和实现状态。
2. 稳定公式同步到 `equations.tex`，保留与主笔记一致的公式编号。
3. 代码中的实现注释引用公式编号，例如 `OBS-06`，不要在代码中维护另一套推导。
4. 论文从 `.tex` 复制；PPT 从 `.tex` 复制到 PowerPoint 公式框；汇报叙述从 `.md` 提炼。
5. 实验数值只写实测结果，不用预期结果替代；附运行命令和 CSV 路径。

## 状态标记

- **[Implemented]**：代码中已经实现并有测试覆盖。
- **[Partial]**：已有骨架或只在特定假设下实现。
- **[Planned]**：设计已明确但尚未完成。
- **[Question]**：仍需理论核对或实验确认。
