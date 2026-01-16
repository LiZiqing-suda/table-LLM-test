# **CRT-QA**

现实场景中，用户针对表格提出的问题往往具有隐含性，甚至是模糊不清的。（无法回答或答案不确定的查询样本）

该论文将“推理”定义为“非形式化推理”，即需要通过直觉、经验、常识来推导结论和解决问题。以往的研究将“推理”定义为以**筛选** 操作为代表的基础操作，而本文将其称为 **“操作”** 。

针对表格分析中常用的推理类型，构建了一套细粒度的分类体系。见表1。

<img width="794" height="619" alt="image" src="https://github.com/user-attachments/assets/a484cd94-7e8c-49eb-a873-11241bd98075" />

## **数据选取**

从TabFact数据集中选取开放域表格，该数据集的表格均源自维基百科。

## **初始问题生成**

使用大模型作为问题生成器，设计了一条包含问题生成要求的指令型提示词，用于生成候选问题。

使用提示词驱动大模型生成问题有以下缺陷：（1）**复杂度不足：** 已经在提示词中提出了对问题复杂度的要求，但是生成的问题仍然是不包含多跳推理的简单问题；（2）**多样性匮乏：** 当要求 ChatGPT 生成多个问题时，我们发现大量查询语句的格式高度相似，例如，多数问题均以 “是否存在” 开头；（3）**无效问题：** 可能生成仅依靠现有表格无法回答的问题。

人工筛选与反馈：让人工标注员筛选出符合要求的问题，再向大型语言模型提供反馈，以此提升问题的质量。在反馈设计环节，我们采用了若干词汇特征（例如 “运用数学知识”“提升复杂度” 等），用于解决前文提及的各类问题，并降低潜在偏差。**也就是说，指令更具体，大模型生成的问题质量越高。**

<img width="1015" height="561" alt="image" src="https://github.com/user-attachments/assets/c15fb29a-51fd-44b6-9b6f-0ecea52eb3b0" />

## **细粒度标注**

**问题直接性：** 若一个问题仅通过问题本身的词汇、其词形变化形式以及功能词即可表述，则该问题为显性问题；而隐性问题需要借助新的实义词来描述推理过程。

**分解类型：** 

（1）**桥接型：** 需要先找到第一跳证据，才能进一步获取第二跳证据；

（2）**交集型：** 需要找到同时满足两个独立条件的实体；

（3）**对比型：** 需要对两个不同实体的属性特征进行比较。

**人类推理路径：**

由于数据集太大了，所以先设计一套模板填充方式。

具体流程为：首先，标注人员按顺序为每一步骤选定对应的推理或操作类型；随后，针对每个步骤，在预设模板中填入该类型对应的操作目标。例如，若某个查询的解答涉及聚合操作，标注人员需要选定聚合的具体类型（如求和）及其操作对象（如列名）。

# **ARC方法**

该论文还提出了ARC方法，也就是使用代码进行计算。具体就是让模型根据用户的自然语言描述先生成python代码再运行代码来推理结果，避免因为大模型数值计算能力差导致错误。

# **数据集格式**

**针对该csv样例的第一个问题实例：**

step1	操作类 - **索引**，针对rank字段	第一步操作：定位表格中的rank（排名）列，为后续筛选做准备

step2	操作类 - **筛选**，针对rank字段	第二步操作：根据rank列筛选出排名前 5的郡数据（这一步隐含了 “top 5” 的约束，是回答问题的前提）

step3	推理类 - **聚合**，计算AVG（均值），时间范围 1960-2040	第三步推理：针对筛选后的前 5 个郡，提取它们 1960 到 2040 年的人口变化百分比数据

step4	推理类 - **聚合**，计算AVG（均值），目标为人口变化百分比	第四步推理：对前 5 个郡的人口变化百分比数据，计算最终的平均值

Directness	Explicit	问题属于**显性**问题，意味着答案可以通过对表格数据的操作和计算直接得到，不需要额外的外部知识补充

Composition Type	Bridging	问题属于**桥接型**推理，需要先完成 “筛选前 5 郡” 的步骤，再基于该结果进行 “计算均值” 的步骤，两步之间存在依赖桥接关系，无法跳过前序步骤直接计算



    "2-1064198-3.html.csv": [
        {
            "Question name": "What is the average percentage change in population for the top 5 ranked Norwegian counties between 1960 and 2040?",
            "Tittle": "ranked list of norwegian counties",
            "step1": {
                "type": "Operation",
                "name": "Indexing",
                "detail": "rank"
            },
            "step2": {
                "type": "Operation",
                "name": "Filter",
                "detail": "rank"
            },
            "step3": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "AVG",
                    "between 1960 and 2040"
                ]
            },
            "step4": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "AVG",
                    "average percentage change"
                ]
            },
            "Answer": "9.173",
            "Directness": "Explicit",
            "Composition Type": "Bridging"
        },
        {
            "Question name": "Which county has had the most consistent percentage change in population over the three time periods?",
            "Tittle": "ranked list of norwegian counties",
            "step1": {
                "type": "Reasoning",
                "name": "Other Commonsense Reasoning",
                "detail": "consistent percentage change"
            },
            "step2": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "STD",
                    "percentage change"
                ]
            },
            "step3": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "MIN",
                    "percentage change"
                ]
            },
            "step4": {
                "type": "Operation",
                "name": "Indexing",
                "detail": "county"
            },
            "Answer": "norway",
            "Directness": "Explicit",
            "Composition Type": "Bridging"
        },
        {
            "Question name": "Which county has had the fastest rate of population growth between 1960 and 2040, in terms of percentage change per decade?",
            "Tittle": "ranked list of norwegian counties",
            "step1": {
                "type": "Reasoning",
                "name": "Arithmetic",
                "detail": [
                    "-",
                    "1960 and 2040"
                ]
            },
            "step2": {
                "type": "Reasoning",
                "name": "Other Commonsense Reasoning",
                "detail": "fastest"
            },
            "step3": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "MAX",
                    "rate of population growth"
                ]
            },
            "step4": {
                "type": "Operation",
                "name": "Indexing",
                "detail": "county"
            },
            "Answer": "akershus",
            "Directness": "Explicit",
            "Composition Type": "Bridging"
        },
        {
            "Question name": "What is the percentage change in population for the county with the smallest population in 2040, relative to its population in 2000?",
            "Tittle": "ranked list of norwegian counties",
            "step1": {
                "type": "Operation",
                "name": "Indexing",
                "detail": "% (2040)"
            },
            "step2": {
                "type": "Reasoning",
                "name": "Aggregating",
                "detail": [
                    "MIN",
                    "% (2040)"
                ]
            },
            "step3": {
                "type": "Operation",
                "name": "Indexing",
                "detail": "county"
            },
            "step4": {
                "type": "Reasoning",
                "name": "Arithmetic",
                "detail": [
                    "-",
                    "2040, relative to its population in 2000"
                ]
            },
            "Answer": "0.4%",
            "Directness": "Explicit",
            "Composition Type": "Comparison"
        }
    ]


## 表格：

        rank#county#% (1960)#% (2000)#% (2040)
        1#oslo#13.2#11.3#12.8
        2#akershus#6.3#10.4#11.9
        3#hordaland#9.4#9.7#10.2
        4#rogaland#6.6#8.3#9.9
        5#sør - trøndelag#5.8#5.8#6.0
        6#østfold#5.6#5.5#5.5
        7#buskerud#4.6#5.2#5.4
        8#møre og romsdal#5.9#5.4#4.8
        9#nordland#6.6#5.3#3.9
        10#vestfold#4.8#4.7#4.7
        11#hedmark#4.9#4.1#3.4
        12#oppland#4.6#4.0#3.3
        13#vest - agder#3.0#3.4#3.6
        14#telemark#4.1#3.6#3.0
        15#troms#3.5#3.3#2.7
        16#nord - trøndelag#3.2#2.8#2.4
        17#aust - agder#2.1#2.2#2.3
        18#sogn og fjordane#2.8#2.4#1.8
        19#finnmark#2.0#1.6#1.2
        sum#norway#100.0#100.0#100.0
        <img width="91" height="505" alt="image" src="https://github.com/user-attachments/assets/9d01d626-7a16-425c-b12f-004751bf48e5" />
