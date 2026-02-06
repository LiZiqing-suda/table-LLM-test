# **自己做的表格理解数据集。**

爬了ICLR2024和2025所有挂在arxiv并且有html版本的论文。把他们的html文件爬下来并且用python代码对三线表进行了提取。数据集制作的中间结果存在tmpdataset目录下。中间结果不一定准确。

**数据集格式：**

    （1）每个以v1、v2这种版本号结尾的zip文件是一整个数据集。每加工处理一次就提交一个新的版本上来。
    
    （2）每个数据集里面有若干个目录，以arxiv号命名，每个目录对应一篇论文的提取结果。
    
    （3）每个以arxiv号命名的目录下有一个captions.txt文件，记录表格的标题名称。然后有多个csv文件，是表格的内容。表格的编号是按html里面的
    
    先后顺序进行的编号，正常情况下也就是pdf文件当中的先后顺序。还有多个txt文件，除了captions.txt以外均为论文中引用到这些表格的语句，一
    
    行一句。如Table_1.txt为所有引用到Table 1的句子。还有个多表引用句子的txt，为Multi-table.txt。每个句子只会在一个txt文件中出现。

    （4）csv的命名方式是Table_num.csv（如Table_2.csv、Table_14.csv），

**v1：** 初步进行了提取，还没有做验证和整理，仅提取了表格内容及其对应的标题。表格内容可能有因为html渲染语法的原因出现少数提取错误的情况，后续需人工筛选验证。暂未对文章引用这些表格的语句进行提取。

**v2：** 对html内所有引用到这些表格的语句进行了提取。具体提取的内容是：从提到该表格的句子开始，到这一段结束，均归类到该表格对应语句的txt内。如果提到多表格，则优先放在Multi-table.txt。每个句子只会在一个txt文件中出现。且表格新增了markdown格式。

如It is difficult for non-expert users to assess the accuracy of the generated code, we automatically utilize the Example information to verify the accuracy of the CoNN model - checking whether the output result of the input sequence is exactly consistent with the Example. The results shown in **Table 4** demonstrate that generally 2 Examples are sufficient to select an accurate CoNN model, which means it is very easy for users to use and demonstrate. However, considering the varying difficulty of different tasks, we still suggest non-expert users provide more Examples to ensure the accuracy of the generated CoNN.

这段话提取的内容是The results shown in **Table 4** demonstrate that generally 2 Examples are sufficient to select an accurate CoNN model, which means it is very easy for users to use and demonstrate. However, considering the varying difficulty of different tasks, we still suggest non-expert users provide more Examples to ensure the accuracy of the generated CoNN.

存放在Table_4.txt内。

**v2信息统计：**

        参与统计的论文文件夹数量：1556
        参与统计的表格总数：7915
        有效引用句总数：10251
        --------------------------------------------------------------------------------
        
        【一、表格引用句数分布统计】
        各引用句数对应的表格数量（按句数升序）：
          被1句引用的表格：6277个（占比79.31%）
          被2句引用的表格：1210个（占比15.29%）
          被3句引用的表格：270个（占比3.41%）
          被4句引用的表格：100个（占比1.26%）
          被5句引用的表格：30个（占比0.38%）
          被6句引用的表格：14个（占比0.18%）
          被7句引用的表格：8个（占比0.1%）
          被8句引用的表格：2个（占比0.03%）
          被9句引用的表格：2个（占比0.03%）
          被10句引用的表格：2个（占比0.03%）
        --------------------------------------------------------------------------------
        
        【二、引用句单词数分布统计】
        1. 单词数基本统计量：
          总句数：10251
          最小值：3
          最大值：502
          平均值：69.39
          中位数：57
          标准差：53.55
        
        2. 单词数区间分布（数量/占比）：
          0-5个单词：39句（占比0.38%）
          6-10个单词：502句（占比4.9%）
          11-15个单词：644句（占比6.28%）
          16-20个单词：539句（占比5.26%）
          21以上个单词：8527句（占比83.18%）

## **引用句子分级**

        1. S级（优质）：直接聚焦表格内容，包含对比/数值/趋势/结论，可直接作为事实验证句/QA问题原型（如：Model A 比 Model B 准确率高 3.2%）。所有需要修改内容和清理无关括号或分句才能用作事实验证和QA问题素材的都不能是S级。
        2. A级（有效）：聚焦表格内容，无无关信息，但无具体对比/数值（如：LoRA+RLHF 是所有微调策略中效果最好的）
        3. B级（弱相关）：核心内容与表格相关，但夹杂无关信息（如结合论文其他实验、方法描述），或表述模糊（如：Table 2 展示了我们的实验结果，该结果支撑了本文的核心论点）
        4. C级（无效）：仅提及表格编号，无任何与表格内容相关的信息（如：Table 2 的结果见下文分析、我们在 Table 2 中报告了相关数据）

调用gpt-5-mini模型（这一步并不涉及表格精确定位和多步推理的任务，mini版足够）的api对引用的句子进行分级，输入的表格格式为markdown格式，更方便模型对表格进行理解。结果为table_citation_grade.jsonl。

再使用开源模型计算困惑度，对分级结果进行验证。
