---
ars_mode: academic-paper/full
target_venue: EMNLP-2026 (Findings Track)
style_format: LaTeX / apa7 document class (manuscript mode)
control_variables: ARS_CLAIM_AUDIT=1, ARS_CROSS_MODEL=1
---

# 1. 论文核心定位与科学叙事铁律 (Linguistic Paradigm Split)
本论文拒绝走传统的“方言转写报错率对比”工程流水账，而是定位为：【语音基座模型在面对非标准方言连续体时的解码行为学实证解构】。
你必须在全文（Intro, Results, Discussion）中死死咬住以下由真实实验数据（Empirical Ground Truth）支撑的三大发现，将其升华为论文的“Aha! Moment”：

1.1 纯声学 CTC 的“声学诚实性” (Acoustic Honesty of CTC):
   - 数据事实：wav2vec_1b 在 SwissDial 上的 DPS 为正值 (+0.0649)。
   - 科学叙事：纯声学模型由于缺乏强力的语言模型（LM）文本解码器，被迫保持了对原始音位的忠实捕捉。它们听到方言音，就倔强地用字符硬拼（Phonetic Mapping），从而抵抗了标准语的吸引力。

1.2 生成式模型的“容量先验霸权” (Vocabulary-Driven Prior Erosion):
   - 数据事实：Whisper 家族表现出单调的负向漂移。在 Betthupferl 上，随着模型规模扩大，DPS 从 Tiny (-0.10) 单调下滑至 Large-v3 (-0.18)。
   - 科学叙事：模型容量（Scale）的激增并没有民主化低资源方言的识别，反而加剧了高资源语言的流形霸权。参数量越大，自回归解码器在低信噪比/干净音频下，越倾向于用庞大的高地德语先验“反客为主”重写（Rewrite）声学证据，强行抹杀方言变体（Default to Standards）。

1.3 融合大模型 ASR 的“语义包容混合语” (The Pseudo-Equilibrium of LLM-ASRs):
   - 数据事实：Qwen3-ASR-0.6B / 1.7B 的 DPS 极度趋近于 0 轴（-0.0019 / -0.0144），但其绝对 NormSim 和 DialectSim 双双突破 0.81-0.84。
   - 科学叙事：这绝不代表大模型“没有偏见”，而是一种【高绝对值下的伪平衡】。LLM-ASR 凭借超强的上下文包容力，生成了一种独特的“混合语（Hybrid Register）”——它在句法（Syntax）和词序上完美保留了方言骨架（使 DialectSim 飙高），但在具体的字形（Orthography）上采用了标准语拼写（使 NormSim 飙高）。它成功打破了方言保留与标准规范的“零和博弈”。

# 2. 严苛的方法论描述与数学锚定 (Methodology Safeguards)
为了通过 ARS v3.8.0 的 FNR/FPR 审计闸门，Methodology 部分必须具备像素级数学严谨性：
- 显式写出基于归一化字符编辑距离（NCED）的 DialectSim、NormSim 和 DPS 联动公式。
- 【零值保护门槛（Degenerate Case Guardrail）】：必须写入防御条款——当 max(DialectSim, NormSim) < tau (tau=0.1) 时，判定为极端噪声下的模型完全崩溃（Garbage Output），由代码自动 Drop 过滤，以消除生存者偏差（Survival Bias），确保低 SNR 处的收敛不是因为数值对撞归零。

# 3. 文献支撑与跨领域学术锚定 (The Trust-Chain Citation Frontmatter)
根据 ARS 审计要求，你必须将本研究的音频实证结论与以下四个领域的顶级文本偏见文献进行 claim-level 的深度绑定：
- 引入 Fleisig et al. (2024) [Linguistic Bias in ChatGPT] 提出的“标准语默认（Default to Standard）”概念，作为 Whisper 先验塌陷的理论先驱。
- 引入 Naous et al. (2024) [CAMEL / Measuring Cultural Bias] 关于大模型多语言语义空间向高资源流形倾斜的发现，以此解释 Qwen3 混合语字形标准化的成因。
- 引用 Cotterell et al. [Are All Languages Equally Hard to Language-Model] 关于复杂形态/非标准文本在语言学熵增上的限制，解释 Wav2Vec2 坚守声学诚实的代价与本质。
- 引用 Deas et al. [African American Language Bias] 的二元评估困境，论证本论文发明双参考锚点 DPS 指标的方法学正当性（Justification）。

# 4. 数据隔离与异质性防御铁律 (Dataset Heterogeneity Defended)
- 严禁将 SwissDial（高质量 Parallel 对齐）和 Betthupferl（非规范历史方言组）的绝对 $DPS$ 数值进行跨库平均（Cross-pool Aggregation）。
- 必须将 Betthupferl 诚实地声明为【相对排序验证集（Relative Rank-Order Validator）】。将其整体下移的负向趋势解释为“历史标注语料天然具备的高地德语规范化偏见（Annotation Artifact）”，变审稿人可能攻击的“数据断层地雷”为我方体现“科学诚实性”的学术亮点。