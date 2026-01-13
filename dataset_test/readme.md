# **对现在的表格理解benchmark数据集进行汇总整理，单表推理部分**

## （1）TabFact（ICLR2020）

**任务类型：** 表格事实验证。每个问题只能回答正确或错误，没有“对一半”“不严谨”这种情况。

**核心特色和解决的问题：** 117,854 条人工标注语句与16,573 个 Wikipedia 表格。html格式，含行列结构与单元格内容。整个数据集分为了简单任务和复杂任务两个部分，简单任务只涉及单行就能回答，如三月南京平均气温是否为5度这种；复杂任务需要多行聚合、比较，需要模型定位多行内容和较强的数值计算能力。

**存在的问题：** 某些问题不能简单用“正确错误”评判，还有一些较复杂的问题容易引起歧义。且数据缺乏真实世界噪声。

## （2）TAT-QA（ACL2021）

**任务类型：** 金融领域表格 + 文本混合 QA（含数值计算推理）

**核心特色和解决的问题：** 16552个问答对。混合上下文，不只是表格数据，还有纯文本作为数据，解决纯表格QA无法解决的跨模态信息融合的问题。涵盖数值计算、比较、排序、比例计算等复杂任务，对模型数值计算能力要求高。每个样本含1个半结构化表格+至少2段关联文本，金融专业术语多，数据来源真实且存在噪声。分为三个难度，分别是（1）单步计算，只需要表格数据；（2）多步计算，或需表格 + 文本单一来源；（3）多步计算且需要文本表格跨模态融合。答案类型是**精确数值**。

**存在的问题：** 全是金融领域问题，泛化能力弱。

    {
        "table": {
          "uid": "c4b92833-5c85-4bf4-b493-bc7741d759df",
          "table": [
            [
              "",
              "Year Ended",
              "Year Ended"
            ],
            [
              "Stock-Based Compensation by Type of Award",
              "December 31, 2019",
              "December 31, 2018"
            ],
            [
              "Stock options",
              "$2,756",
              "$2,926"
            ],
            [
              "RSUs",
              "955",
              "1,129"
            ],
            [
              "Total stock-based compensation expense",
              "$3,711",
              "$4,055"
            ]
          ]
        },
        "paragraphs": [
          {
            "uid": "04bfbe1d-235b-4036-95c2-e49983eb9cef",
            "order": 1,
            "text": "Stock-based compensation expense is included in general and administrative expense for each period as follows:"
          },
          {
            "uid": "0b5304d0-849b-46ea-936a-2b9d73be07f3",
            "order": 2,
            "text": "As of December 31, 2019, there was $4,801 of unrecognized stock-based compensation expense related to unvested employee stock options and $1,882 of unrecognized stock-based compensation expense related to unvested RSUs. These costs are expected to be recognized over a weighted-average period of 2.13 and 2.33 years, respectively."
          }
        ],
        "questions": [
          {
            "uid": "7c884c23-7774-4414-b817-d41dd797319b",
            "order": 1,
            "question": "What was the amount of unrecognized stock-based compensation expense related to unvested employee stock options in 2019?"
          },
          {
            "uid": "53f1517b-bdd8-4165-8adb-0aafadbf0588",
            "order": 2,
            "question": "What was the total stock-based compensation expense amount in 2018?"
          },
          {
            "uid": "65c5aed2-6ce6-4ac7-ad25-713d2bf3ead4",
            "order": 3,
            "question": "How long is it expected to take for the unrecognized stock-based compensation expense related to unvested RSUs to be recognized?"
          },
          {
            "uid": "15cfd097-55b4-4e9f-847e-bc78e8e35136",
            "order": 4,
            "question": "What is the total stock-based compensation expense and unrecognized stock-based compensation expense in 2019?"
          },
          {
            "uid": "a4dfd2d1-4fa0-4fd7-a1d3-889bb36489a4",
            "order": 5,
            "question": "What was the change in the amount of stock options in 2019 from 2018?"
          },
          {
            "uid": "208f5e40-e37c-4aaa-b9d7-74d148f39c75",
            "order": 6,
            "question": "What was the percentage change in the amount of RSUs in 2019 from 2018?"
          }
        ]
      }

## （3）SEM-TAB-FACTS（2021）

**任务类型：** 科学文档表格事实验证和证据定位。

**核心特色和解决的问题：** 面向科学文献表格的事实验证基准，包含981 个人工生成表格与1980 个自动生成表格。双任务评估：A 任务（事实分类：支持 / 反驳 / 未知）+ B 任务（证据定位：精确到单元格）。

数据形式：表格+陈述+标签+证据+领域标签。

**存在的问题：** 人工生成表格少，自动生成表格可能存在质量问题。

## （4）KaggleDBQA（ACL2021）

**任务类型：** 真实数据库文本转SQL+表格QA。

**核心特色和解决的问题：** 基于真实的Kaggle数据库，包含12个领域24个数据库10347个问答对。引入了噪声，解决之前数据集过于理想化的问题。包含原始数据库文件（表结构、索引、约束）、数据库说明文档（各字段含义）、问题（自然语言查询）、SQL（标准SQL查询语句，我们可以不用他）、答案（表格或单值）。

**存在的问题：** 主要是数据库SQL查询任务，由于SQL查询可能会出现查完仍然是一个表的情况，我们只能使用该数据集答案为单值的样例，或仅使用数据库表格而不用他的问题和答案。

## （5）PubHealthTab（NAACL2022）

**任务类型：** 公共卫生领域表格事实验证。

**核心特色和解决的问题：** 模型需要判断一个自然语言声明（claim）是否被一个表格证据支持（supports）、反驳（refutes），或者表格中没有足够的信息（NEI，Not Enough Information）来判断。

**存在的问题：** 类别不平衡，支持类过多（超过50%），数值计算任务过多对模型数值计算能力要求很高，且领域集中在公共卫生领域，适用范围窄。

## （6）FEVEROUS（EMNLP2022）

**任务类型：** 多模态（表格和文本）事实验证。

**核心特色和解决的问题：** 包含87026个声明与18840个表格标注过程要求同时验证文本与表格证据，解决跨模态信息冲突问题，给模型提供文本和表格冲突时的采纳优先级依据。包含“支持”“反对”“找不到”三种结果。

**存在的问题：** 简单表格为主，缺乏层级表格和合并单元格。**我认为结果的这三种分类是不恰当的，对于跨模态信息冲突，应该使用第四种结果，也就是“矛盾”，而不是强行让模型学习一种强行解释的逻辑。** 因为很多矛盾的情况都是使用者写错了，可能是表格写错了也可能是文本写错了，没有一个固定逻辑。

## （7）HiTab（ACL2022）

**任务类型：** 层次化表格QA。

**核心特色和解决的问题：** 10,686 个 QA 对与3,597 个表格，很多层次化表格，数据来自真实场景。解决传统表格QA无法处理的层次结构理解问题。

**存在的问题：** 表格过于复杂，标注难度很大，存在一些标注错误情况。

## （8）IM-TQA（ACL2023）

**任务类型：** 中文隐式结构表格 QA和结构理解。

**核心特色和解决的问题：** 中文数据集，数据来自公开网页，覆盖多领域。**推理阶段不直接提供表头标注，模型需自主识别表头与数据单元格。** 主要用于真实场景中大量无标注的表格推理的情况，比如说日常生活中使用大模型进行表格理解，普通用户根本不可能去输入什么“TAB”“CELL”这种标记token，最多就是用空格和换行去间隔不同单元格。这就需要模型自己识别表格结构。除了表格QA，该数据集还可以用于表格结构理解任务。

**存在的问题：** 数值计算难度较低，数据集规模较小，只有中文没有英文，而且复杂表格情况较少。

## （9）LongTableBench（EMNLP2025）

**任务类型：** 长上下文表格推理。

**核心特色和解决的问题：** 超长表格推理，最长128Ktokens。支持7种表格形式，覆盖18个领域。还有多轮对话和多表联合场景。生成自然语言答案而不是固定数值或文本答案。

**存在的问题：** 表格太长了，训练不动。答案是生成式回答，效果不好评估，且有多轮问答，进一步加大了难度。

## （10）TableEval（EMNLP2025）

**任务类型：** 复杂多语言多结构表格QA。

**核心特色和解决的问题：** 多种语言和多种表格结构，表格覆盖领域较多。

**存在的问题：** 同样是生成式答案，不适合我们的工作。

## （11）MMTU（2025）

**任务类型：** 大规模多任务表格理解。任务过多，9大类任务。pass
