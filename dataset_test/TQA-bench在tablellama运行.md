## **TQA-bench研究本身已经用了tablellama模型进行了测试，本文进行一下总结**

由于tablellama模型最大只能支持8K上下文长度，所以论文只测试了8K上下文长度的数据。

8K 上下文长度（模型理论支持的最大长度）：结果标记为 “NFI”（Not Following Instructions，不遵循指令）。这意味着 TableLlama 在处理 8K 上下文的 TQA-Bench 任务时，无法按照评估要求的格式（指定格式 “Answer: A/B/C/D”）生成答案，例如可能输出无关内容或不满足任务指令。
