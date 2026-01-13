# **对现在的表格理解benchmark数据集进行汇总整理，单表推理部分**

## （1）TabFact（ICLR2020）

**任务类型：** 表格事实验证。每个问题只能回答正确或错误，没有“对一半”“不严谨”这种情况。

**核心特色和解决的问题：** 117,854 条人工标注语句与16,573 个 Wikipedia 表格。html格式，含行列结构与单元格内容。整个数据集分为了简单任务和复杂任务两个部分，简单任务只涉及单行就能回答，如三月南京平均气温是否为5度这种；复杂任务需要多行聚合、比较，需要模型定位多行内容和较强的数值计算能力。

**存在的问题：** 某些问题不能简单用“正确错误”评判，还有一些较复杂的问题容易引起歧义。且数据缺乏真实世界噪声。

    {
      "instruction": "This is a table fact verification task. The goal of this task is to distinguish whether the given statement is entailed or refuted by the given table.",
      "input": "[TLE] The table caption is about tony lema. [TAB] | tournament | wins | top - 5 | top - 10 | top - 25 | events | cuts made [SEP] | masters tournament | 0 | 1 | 2 | 4 | 4 | 4 | [SEP] | us open | 0 | 2 | 3 | 4 | 6 | 5 | [SEP] | the open championship | 1 | 2 | 2 | 2 | 3 | 3 | [SEP] | pga championship | 0 | 0 | 1 | 2 | 5 | 4 | [SEP] | totals | 1 | 5 | 8 | 12 | 18 | 16 |",
      "question": "The statement is:  <tony lema be in the top 5 for the master tournament , the us open , and the open championship>. Is it entailed or refuted by the table above?",
      "output": "entailed",
      "input_seg": "[TLE] The table caption is about tony lema. [TAB] | tournament | wins | top - 5 | top - 10 | top - 25 | events | cuts made [SEP] | masters tournament | 0 | 1 | 2 | 4 | 4 | 4 | [SEP] | us open | 0 | 2 | 3 | 4 | 6 | 5 | [SEP] | the open championship | 1 | 2 | 2 | 2 | 3 | 3 | [SEP] | pga championship | 0 | 0 | 1 | 2 | 5 | 4 | [SEP] | totals | 1 | 5 | 8 | 12 | 18 | 16 |"
    }

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

        <document>
        <table id="Table 2">
        <caption text=" Networks across East Asia. "> </caption>
        <legend text=" Data are for 1290 firms across nine East Asian economies. All network data are assembled by the authors, and are cross-sectional for 2008. Table reports country-level statistics on board networks, family networks, state networks, and political networks. Minimum values are everywhere 0. board network counts the amount of board/executive interlocks. Political network counts the amount of board/executive interlocks with politically-connected firms. Family network counts the amount of board/executive interlocks with family-controlled firms. State network counts the amount of board/executive interlocks with state-owned firms. "> </legend>
        <row row="0">
        <cell col-end="0" col-start="0" row-end="1" row-start="0" text="Country"> </cell>
        <cell col-end="1" col-start="1" row-end="1" row-start="0" text="N"> </cell>
        <cell col-end="4" col-start="2" row-end="0" row-start="0" text="Board network"> </cell>
        <cell col-end="7" col-start="5" row-end="0" row-start="0" text="Family network"> </cell>
        <cell col-end="10" col-start="8" row-end="0" row-start="0" text="State network"> </cell>
        <cell col-end="13" col-start="11" row-end="0" row-start="0" text="Political network"> </cell>
        </row>
        <row row="1">
        <cell col-end="2" col-start="2" row-end="1" row-start="1" text="mean"> </cell>
        <cell col-end="3" col-start="3" row-end="1" row-start="1" text="SD"> </cell>
        <cell col-end="4" col-start="4" row-end="1" row-start="1" text="max"> </cell>
        <cell col-end="5" col-start="5" row-end="1" row-start="1" text="mean"> </cell>
        <cell col-end="6" col-start="6" row-end="1" row-start="1" text="SD"> </cell>
        <cell col-end="7" col-start="7" row-end="1" row-start="1" text="max"> </cell>
        <cell col-end="8" col-start="8" row-end="1" row-start="1" text="mean"> </cell>
        <cell col-end="9" col-start="9" row-end="1" row-start="1" text="SD"> </cell>
        <cell col-end="10" col-start="10" row-end="1" row-start="1" text="max"> </cell>
        <cell col-end="11" col-start="11" row-end="1" row-start="1" text="mean"> </cell>
        <cell col-end="12" col-start="12" row-end="1" row-start="1" text="SD"> </cell>
        <cell col-end="13" col-start="13" row-end="1" row-start="1" text="max"> </cell>
        </row>
        <row row="2">
        <cell col-end="0" col-start="0" row-end="2" row-start="2" text="Hong Kong"> </cell>
        <cell col-end="1" col-start="1" row-end="2" row-start="2" text="133"> </cell>
        <cell col-end="2" col-start="2" row-end="2" row-start="2" text="5.12"> </cell>
        <cell col-end="3" col-start="3" row-end="2" row-start="2" text="6.1"> </cell>
        <cell col-end="4" col-start="4" row-end="2" row-start="2" text="33"> </cell>
        <cell col-end="5" col-start="5" row-end="2" row-start="2" text="2.62"> </cell>
        <cell col-end="6" col-start="6" row-end="2" row-start="2" text="4.51"> </cell>
        <cell col-end="7" col-start="7" row-end="2" row-start="2" text="26"> </cell>
        <cell col-end="8" col-start="8" row-end="2" row-start="2" text="1.00"> </cell>
        <cell col-end="9" col-start="9" row-end="2" row-start="2" text="1.41"> </cell>
        <cell col-end="10" col-start="10" row-end="2" row-start="2" text="6"> </cell>
        <cell col-end="11" col-start="11" row-end="2" row-start="2" text="0.67"> </cell>
        <cell col-end="12" col-start="12" row-end="2" row-start="2" text="1.37"> </cell>
        <cell col-end="13" col-start="13" row-end="2" row-start="2" text="6"> </cell>
        </row>
        <row row="3">
        <cell col-end="0" col-start="0" row-end="3" row-start="3" text="Indonesia"> </cell>
        <cell col-end="1" col-start="1" row-end="3" row-start="3" text="169"> </cell>
        <cell col-end="2" col-start="2" row-end="3" row-start="3" text="1.64"> </cell>
        <cell col-end="3" col-start="3" row-end="3" row-start="3" text="3.31"> </cell>
        <cell col-end="4" col-start="4" row-end="3" row-start="3" text="23"> </cell>
        <cell col-end="5" col-start="5" row-end="3" row-start="3" text="0.95"> </cell>
        <cell col-end="6" col-start="6" row-end="3" row-start="3" text="2.64"> </cell>
        <cell col-end="7" col-start="7" row-end="3" row-start="3" text="17"> </cell>
        <cell col-end="8" col-start="8" row-end="3" row-start="3" text="0.14"> </cell>
        <cell col-end="9" col-start="9" row-end="3" row-start="3" text="0.38"> </cell>
        <cell col-end="10" col-start="10" row-end="3" row-start="3" text="2"> </cell>
        <cell col-end="11" col-start="11" row-end="3" row-start="3" text="0.22"> </cell>
        <cell col-end="12" col-start="12" row-end="3" row-start="3" text="1.09"> </cell>
        <cell col-end="13" col-start="13" row-end="3" row-start="3" text="9"> </cell>
        </row>
        <row row="4">
        <cell col-end="0" col-start="0" row-end="4" row-start="4" text="Japan"> </cell>
        <cell col-end="1" col-start="1" row-end="4" row-start="4" text="126"> </cell>
        <cell col-end="2" col-start="2" row-end="4" row-start="4" text="1.84"> </cell>
        <cell col-end="3" col-start="3" row-end="4" row-start="4" text="2.33"> </cell>
        <cell col-end="4" col-start="4" row-end="4" row-start="4" text="15"> </cell>
        <cell col-end="5" col-start="5" row-end="4" row-start="4" text="0.07"> </cell>
        <cell col-end="6" col-start="6" row-end="4" row-start="4" text="0.42"> </cell>
        <cell col-end="7" col-start="7" row-end="4" row-start="4" text="3"> </cell>
        <cell col-end="8" col-start="8" row-end="4" row-start="4" text="0.09"> </cell>
        <cell col-end="9" col-start="9" row-end="4" row-start="4" text="0.31"> </cell>
        <cell col-end="10" col-start="10" row-end="4" row-start="4" text="2"> </cell>
        <cell col-end="11" col-start="11" row-end="4" row-start="4" text="0.00"> </cell>
        <cell col-end="12" col-start="12" row-end="4" row-start="4" text="0.00"> </cell>
        <cell col-end="13" col-start="13" row-end="4" row-start="4" text="0"> </cell>
        </row>
        <row row="5">
        <cell col-end="0" col-start="0" row-end="5" row-start="5" text="South Korea"> </cell>
        <cell col-end="1" col-start="1" row-end="5" row-start="5" text="133"> </cell>
        <cell col-end="2" col-start="2" row-end="5" row-start="5" text="2.5"> </cell>
        <cell col-end="3" col-start="3" row-end="5" row-start="5" text="2.8"> </cell>
        <cell col-end="4" col-start="4" row-end="5" row-start="5" text="21"> </cell>
        <cell col-end="5" col-start="5" row-end="5" row-start="5" text="1.09"> </cell>
        <cell col-end="6" col-start="6" row-end="5" row-start="5" text="1.37"> </cell>
        <cell col-end="7" col-start="7" row-end="5" row-start="5" text="6"> </cell>
        <cell col-end="8" col-start="8" row-end="5" row-start="5" text="0.15"> </cell>
        <cell col-end="9" col-start="9" row-end="5" row-start="5" text="0.40"> </cell>
        <cell col-end="10" col-start="10" row-end="5" row-start="5" text="2"> </cell>
        <cell col-end="11" col-start="11" row-end="5" row-start="5" text="0.02"> </cell>
        <cell col-end="12" col-start="12" row-end="5" row-start="5" text="0.15"> </cell>
        <cell col-end="13" col-start="13" row-end="5" row-start="5" text="1"> </cell>
        </row>
        <row row="6">
        <cell col-end="0" col-start="0" row-end="6" row-start="6" text="Malaysia"> </cell>
        <cell col-end="1" col-start="1" row-end="6" row-start="6" text="281"> </cell>
        <cell col-end="2" col-start="2" row-end="6" row-start="6" text="7.35"> </cell>
        <cell col-end="3" col-start="3" row-end="6" row-start="6" text="6.61"> </cell>
        <cell col-end="4" col-start="4" row-end="6" row-start="6" text="37"> </cell>
        <cell col-end="5" col-start="5" row-end="6" row-start="6" text="1.07"> </cell>
        <cell col-end="6" col-start="6" row-end="6" row-start="6" text="1.94"> </cell>
        <cell col-end="7" col-start="7" row-end="6" row-start="6" text="8"> </cell>
        <cell col-end="8" col-start="8" row-end="6" row-start="6" text="2.15"> </cell>
        <cell col-end="9" col-start="9" row-end="6" row-start="6" text="3.09"> </cell>
        <cell col-end="10" col-start="10" row-end="6" row-start="6" text="18"> </cell>
        <cell col-end="11" col-start="11" row-end="6" row-start="6" text="0.36"> </cell>
        <cell col-end="12" col-start="12" row-end="6" row-start="6" text="0.74"> </cell>
        <cell col-end="13" col-start="13" row-end="6" row-start="6" text="5"> </cell>
        </row>
        <row row="7">
        <cell col-end="0" col-start="0" row-end="7" row-start="7" text="Philippines"> </cell>
        <cell col-end="1" col-start="1" row-end="7" row-start="7" text="98"> </cell>
        <cell col-end="2" col-start="2" row-end="7" row-start="7" text="8.52"> </cell>
        <cell col-end="3" col-start="3" row-end="7" row-start="7" text="8.91"> </cell>
        <cell col-end="4" col-start="4" row-end="7" row-start="7" text="38"> </cell>
        <cell col-end="5" col-start="5" row-end="7" row-start="7" text="5.33"> </cell>
        <cell col-end="6" col-start="6" row-end="7" row-start="7" text="6.16"> </cell>
        <cell col-end="7" col-start="7" row-end="7" row-start="7" text="21"> </cell>
        <cell col-end="8" col-start="8" row-end="7" row-start="7" text="0.71"> </cell>
        <cell col-end="9" col-start="9" row-end="7" row-start="7" text="1.59"> </cell>
        <cell col-end="10" col-start="10" row-end="7" row-start="7" text="10"> </cell>
        <cell col-end="11" col-start="11" row-end="7" row-start="7" text="0.20"> </cell>
        <cell col-end="12" col-start="12" row-end="7" row-start="7" text="0.81"> </cell>
        <cell col-end="13" col-start="13" row-end="7" row-start="7" text="6"> </cell>
        </row>
        <row row="8">
        <cell col-end="0" col-start="0" row-end="8" row-start="8" text="Singapore"> </cell>
        <cell col-end="1" col-start="1" row-end="8" row-start="8" text="116"> </cell>
        <cell col-end="2" col-start="2" row-end="8" row-start="8" text="3.52"> </cell>
        <cell col-end="3" col-start="3" row-end="8" row-start="8" text="3.24"> </cell>
        <cell col-end="4" col-start="4" row-end="8" row-start="8" text="15"> </cell>
        <cell col-end="5" col-start="5" row-end="8" row-start="8" text="0.59"> </cell>
        <cell col-end="6" col-start="6" row-end="8" row-start="8" text="1.66"> </cell>
        <cell col-end="7" col-start="7" row-end="8" row-start="8" text="12"> </cell>
        <cell col-end="8" col-start="8" row-end="8" row-start="8" text="1.28"> </cell>
        <cell col-end="9" col-start="9" row-end="8" row-start="8" text="2.40"> </cell>
        <cell col-end="10" col-start="10" row-end="8" row-start="8" text="11"> </cell>
        <cell col-end="11" col-start="11" row-end="8" row-start="8" text="0.57"> </cell>
        <cell col-end="12" col-start="12" row-end="8" row-start="8" text="1.90"> </cell>
        <cell col-end="13" col-start="13" row-end="8" row-start="8" text="14"> </cell>
        </row>
        <row row="9">
        <cell col-end="0" col-start="0" row-end="9" row-start="9" text="Taiwan"> </cell>
        <cell col-end="1" col-start="1" row-end="9" row-start="9" text="107"> </cell>
        <cell col-end="2" col-start="2" row-end="9" row-start="9" text="1.6"> </cell>
        <cell col-end="3" col-start="3" row-end="9" row-start="9" text="2.22"> </cell>
        <cell col-end="4" col-start="4" row-end="9" row-start="9" text="12"> </cell>
        <cell col-end="5" col-start="5" row-end="9" row-start="9" text="0.21"> </cell>
        <cell col-end="6" col-start="6" row-end="9" row-start="9" text="1.11"> </cell>
        <cell col-end="7" col-start="7" row-end="9" row-start="9" text="7"> </cell>
        <cell col-end="8" col-start="8" row-end="9" row-start="9" text="0.14"> </cell>
        <cell col-end="9" col-start="9" row-end="9" row-start="9" text="0.46"> </cell>
        <cell col-end="10" col-start="10" row-end="9" row-start="9" text="3"> </cell>
        <cell col-end="11" col-start="11" row-end="9" row-start="9" text="0.00"> </cell>
        <cell col-end="12" col-start="12" row-end="9" row-start="9" text="0.00"> </cell>
        <cell col-end="13" col-start="13" row-end="9" row-start="9" text="0"> </cell>
        </row>
        <row row="10">
        <cell col-end="0" col-start="0" row-end="10" row-start="10" text="Thailand"> </cell>
        <cell col-end="1" col-start="1" row-end="10" row-start="10" text="127"> </cell>
        <cell col-end="2" col-start="2" row-end="10" row-start="10" text="5.11"> </cell>
        <cell col-end="3" col-start="3" row-end="10" row-start="10" text="5.04"> </cell>
        <cell col-end="4" col-start="4" row-end="10" row-start="10" text="23"> </cell>
        <cell col-end="5" col-start="5" row-end="10" row-start="10" text="1.58"> </cell>
        <cell col-end="6" col-start="6" row-end="10" row-start="10" text="3.15"> </cell>
        <cell col-end="7" col-start="7" row-end="10" row-start="10" text="19"> </cell>
        <cell col-end="8" col-start="8" row-end="10" row-start="10" text="0.73"> </cell>
        <cell col-end="9" col-start="9" row-end="10" row-start="10" text="1.99"> </cell>
        <cell col-end="10" col-start="10" row-end="10" row-start="10" text="11"> </cell>
        <cell col-end="11" col-start="11" row-end="10" row-start="10" text="0.29"> </cell>
        <cell col-end="12" col-start="12" row-end="10" row-start="10" text="1.16"> </cell>
        <cell col-end="13" col-start="13" row-end="10" row-start="10" text="8"> </cell>
        </row>
        <statements>
        <statement id="2" text="At the same time, these networks often occur in tandem at the firm level." type=""> </statement>
        <statement id="3" text="For each network interaction, there is considerable variation both across and within countries." type=""> </statement>
        <statement id="5" text="The n value is same for Hong Kong and Malaysia." type=""> </statement>
        <statement id="8" text="There are 9 different types country in the given table." type=""> </statement>
        </statements>
        </table>
        </document>

## （4）KaggleDBQA（ACL2021）

**任务类型：** 真实数据库文本转SQL+表格QA。

**核心特色和解决的问题：** 基于真实的Kaggle数据库，包含12个领域24个数据库10347个问答对。引入了噪声，解决之前数据集过于理想化的问题。包含原始数据库文件（表结构、索引、约束）、数据库说明文档（各字段含义）、问题（自然语言查询）、SQL（标准SQL查询语句，我们可以不用他）、答案（表格或单值）。

**存在的问题：** 主要是数据库SQL查询任务，由于SQL查询可能会出现查完仍然是一个表的情况，我们只能使用该数据集答案为单值的样例，或仅使用数据库表格而不用他的问题和答案。用处不太大。

## （5）PubHealthTab（NAACL2022）

**任务类型：** 公共卫生领域表格事实验证。

**核心特色和解决的问题：** 模型需要判断一个自然语言声明（claim）是否被一个表格证据支持（supports）、反驳（refutes），或者表格中没有足够的信息（NEI，Not Enough Information）来判断。

**存在的问题：** 类别不平衡，支持类过多（超过50%），数值计算任务过多对模型数值计算能力要求很高，且领域集中在公共卫生领域，适用范围窄。

        {"_id": "6072bd2a000ca92c09d13a72", "claim": "Brazil likely has 12 times more coronavirus cases than official count, study finds.", "label": "NOT ENOUGH INFO", "header_rationale": [1], "table": {"website": "https://www.nytimes.com/interactive/2021/world/covid-cases.html", "website_title": "Coronavirus World Map: Tracking the Global Outbreak - The New York Times", "caption": null, "header_horizontal": ["", "Cases Daily Avg.", "Per 100,000", "14-day change", "Deaths Daily Avg.", "Per 100,000", "Fully Vaccinated"], "header_vertical": [], "rows": [["Seychelles", "142", "145", "+22%", "1.1", "1.17", "69%"], ["Mongolia", "2,306", "72", "+67%", "11.7", "0.36", "53%"], ["Namibia", "1,649", "66", "+81%", "34.1", "1.37", "1%"], ["Colombia", "25,882", "51", "a1%", "582.9", "1.16", "11%"], ["Uruguay", "1,428", "41", "a55%", "29.0", "0.84", "44%"], ["Oman", "2,002", "40", "+30%", "36.7", "0.74", "4%"], ["Argentina", "17,765", "40", "a26%", "467.7", "1.04", "9%"], ["Maldives", "204", "39", "a39%", "0.4", "0.08", "35%"], ["Kuwait", "1,556", "37", "+1%", "7.0", "0.17", "1%"], ["BrazilA ao", "70,381", "33", "+6%", "1,664.1", "0.79", "12%"]], "html_code": "<table style=\"border: 1px solid black;\"><thead><tr><th style=\"border: 1px solid black;\"></th><th style=\"border: 1px solid black;\"><span><strong>Cases</strong></span><br/> Daily Avg.</th><th style=\"border: 1px solid black;\"><span>Per</span><br/> 100,000</th><th style=\"border: 1px solid black;\"><span>14-day</span><br/> change</th><th style=\"border: 1px solid black;\"><span><strong>Deaths</strong></span><br/> Daily Avg.</th><th style=\"border: 1px solid black;\"><span>Per</span><br/> 100,000</th><th style=\"border: 1px solid black;\"><span><strong>Fully</strong></span><br/> <strong>Vaccinated</strong></th></tr></thead><tbody><tr><td style=\"border: 1px solid black;\">Seychelles</td><td style=\"border: 1px solid black;\">142</td><td style=\"border: 1px solid black;\">145</td><td style=\"border: 1px solid black;\"><div><span>+22%</span></div></td><td style=\"border: 1px solid black;\">1.1</td><td style=\"border: 1px solid black;\">1.17</td><td style=\"border: 1px solid black;\"><span>69%</span></td></tr><tr><td style=\"border: 1px solid black;\">Mongolia</td><td style=\"border: 1px solid black;\">2,306</td><td style=\"border: 1px solid black;\">72</td><td style=\"border: 1px solid black;\"><div><span>+67%</span></div></td><td style=\"border: 1px solid black;\">11.7</td><td style=\"border: 1px solid black;\">0.36</td><td style=\"border: 1px solid black;\"><span>53%</span></td></tr><tr><td style=\"border: 1px solid black;\">Namibia</td><td style=\"border: 1px solid black;\">1,649</td><td style=\"border: 1px solid black;\">66</td><td style=\"border: 1px solid black;\"><div><span>+81%</span></div></td><td style=\"border: 1px solid black;\">34.1</td><td style=\"border: 1px solid black;\">1.37</td><td style=\"border: 1px solid black;\"><span>1%</span></td></tr><tr><td style=\"border: 1px solid black;\">Colombia</td><td style=\"border: 1px solid black;\">25,882</td><td style=\"border: 1px solid black;\">51</td><td style=\"border: 1px solid black;\"><div><span>â1%</span></div></td><td style=\"border: 1px solid black;\">582.9</td><td style=\"border: 1px solid black;\">1.16</td><td style=\"border: 1px solid black;\"><span>11%</span></td></tr><tr><td style=\"border: 1px solid black;\">Uruguay</td><td style=\"border: 1px solid black;\">1,428</td><td style=\"border: 1px solid black;\">41</td><td style=\"border: 1px solid black;\"><div><span>â55%</span></div></td><td style=\"border: 1px solid black;\">29.0</td><td style=\"border: 1px solid black;\">0.84</td><td style=\"border: 1px solid black;\"><span>44%</span></td></tr><tr><td style=\"border: 1px solid black;\">Oman</td><td style=\"border: 1px solid black;\">2,002</td><td style=\"border: 1px solid black;\">40</td><td style=\"border: 1px solid black;\"><div><span>+30%</span></div></td><td style=\"border: 1px solid black;\">36.7</td><td style=\"border: 1px solid black;\">0.74</td><td style=\"border: 1px solid black;\"><span>4%</span></td></tr><tr><td style=\"border: 1px solid black;\">Argentina</td><td style=\"border: 1px solid black;\">17,765</td><td style=\"border: 1px solid black;\">40</td><td style=\"border: 1px solid black;\"><div><span>â26%</span></div></td><td style=\"border: 1px solid black;\">467.7</td><td style=\"border: 1px solid black;\">1.04</td><td style=\"border: 1px solid black;\"><span>9%</span></td></tr><tr><td style=\"border: 1px solid black;\">Maldives</td><td style=\"border: 1px solid black;\">204</td><td style=\"border: 1px solid black;\">39</td><td style=\"border: 1px solid black;\"><div><span>â39%</span></div></td><td style=\"border: 1px solid black;\">0.4</td><td style=\"border: 1px solid black;\">0.08</td><td style=\"border: 1px solid black;\"><span>35%</span></td></tr><tr><td style=\"border: 1px solid black;\">Kuwait</td><td style=\"border: 1px solid black;\">1,556</td><td style=\"border: 1px solid black;\">37</td><td style=\"border: 1px solid black;\"><div><span>+1%</span></div></td><td style=\"border: 1px solid black;\">7.0</td><td style=\"border: 1px solid black;\">0.17</td><td style=\"border: 1px solid black;\"><span>1%</span></td></tr><tr><td style=\"border: 1px solid black;\"><a>BrazilÂ âº</a></td><td style=\"border: 1px solid black;\">70,381</td><td style=\"border: 1px solid black;\">33</td><td style=\"border: 1px solid black;\"><div><span>+6%</span></div></td><td style=\"border: 1px solid black;\">1,664.1</td><td style=\"border: 1px solid black;\">0.79</td><td style=\"border: 1px solid black;\"><span>12%</span></td></tr></tbody></table>"}, "initial_claim": "Brazil likely has 12 times more coronavirus cases than official count, study finds."}

## （6）FEVEROUS（EMNLP2022）

**任务类型：** 多模态（表格和文本）事实验证。

**核心特色和解决的问题：** 包含87026个声明与18840个表格标注过程要求同时验证文本与表格证据，解决跨模态信息冲突问题，给模型提供文本和表格冲突时的采纳优先级依据。包含“支持”“反对”“找不到”三种结果。

**存在的问题：** 简单表格为主，缺乏层级表格和合并单元格。**我认为结果的这三种分类是不恰当的，对于跨模态信息冲突，应该使用第四种结果，也就是“矛盾”，而不是强行让模型学习一种强行解释的逻辑。** 因为很多矛盾的情况都是使用者写错了，可能是表格写错了也可能是文本写错了，没有一个固定逻辑。

        {"evidence": [{"content": ["Mukaradeeb_sentence_1", "Mukaradeeb_cell_0_3_1", "Mukaradeeb_cell_0_2_1"], "context": {"Mukaradeeb_sentence_1": ["Mukaradeeb_title"], "Mukaradeeb_cell_0_3_1": ["Mukaradeeb_title", "Mukaradeeb_header_cell_0_3_0", "Mukaradeeb_header_cell_0_0_0"], "Mukaradeeb_cell_0_2_1": ["Mukaradeeb_title", "Mukaradeeb_header_cell_0_2_0", "Mukaradeeb_header_cell_0_0_0"]}}], "id": 71874, "claim": "Mukaradeeb('Wolf's Den') is a city in Iraq near the Syrian border, in the district of Al-Qa'im, province of Al-Anbar.", "label": "SUPPORTS", "annotator_operations": [{"operation": "start", "value": "start", "time": "0"}, {"operation": "Now on", "value": "?search=", "time": "0.962"}, {"operation": "search", "value": "Mukaradeeb", "time": "9.524"}, {"operation": "Now on", "value": "Mukaradeeb", "time": "10.752"}, {"operation": "Highlighting", "value": "Mukaradeeb_sentence_1", "time": "13.998"}, {"operation": "Highlighting", "value": "Mukaradeeb_cell_0_3_1", "time": "28.724"}, {"operation": "Highlighting", "value": "Mukaradeeb_cell_0_2_1", "time": "32.485"}, {"operation": "finish", "value": "finish", "time": "146.465"}], "challenge": "Combining Tables and Text"}


## （7）HiTabQA（ACL2022）

**任务类型：** 层次化表格QA。

**核心特色和解决的问题：** 10,686 个 QA 对与3,597 个表格，很多层次化表格，数据来自真实场景。解决传统表格QA无法处理的层次结构理解问题。

**存在的问题：** 表格过于复杂，标注难度很大，存在一些标注错误情况。

        {
          "table_id": "100",
          "instruction": "This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical.",
          "input": " [TLE] The table caption is agri-food industry sub-groups for workers aged 15 years and over, two agricultural regions of ontario, 2011. [TAB] | sub-groups of the agri-food industry | eastern ontario | eastern ontario | northern ontario | northern ontario | [SEP] | sub-groups of the agri-food industry | french-language workers | other workers | french-language workers | other workers | [SEP] | sub-groups of the agri-food industry | percent | percent | percent | percent | [SEP] | input and service supply | 2.9 | 2.1 | 2.9 | 1.3 | [SEP] | food, beverage, and tobacco processing | 9.7 | 6.0 | 3.0 | 3.3 | [SEP] | food retail and wholesale | 35.3 | 31.3 | 39.1 | 37.3 | [SEP] | food service | 52.1 | 60.6 | 55.0 | 58.1 |",
          "question": "in eastern ontario, what percent of french-language workers have worked in the restaurant and food services sector?",
          "output": "52.1",
          "raw_answer": [
            52.1
          ],
          "input_seg": " [TLE] The table caption is agri-food industry sub-groups for workers aged 15 years and over, two agricultural regions of ontario, 2011. [TAB] | sub-groups of the agri-food industry | eastern ontario | eastern ontario | northern ontario | northern ontario | [SEP] | sub-groups of the agri-food industry | french-language workers | other workers | french-language workers | other workers | [SEP] | sub-groups of the agri-food industry | percent | percent | percent | percent | [SEP] | input and service supply | 2.9 | 2.1 | 2.9 | 1.3 | [SEP] | food, beverage, and tobacco processing | 9.7 | 6.0 | 3.0 | 3.3 | [SEP] | food retail and wholesale | 35.3 | 31.3 | 39.1 | 37.3 | [SEP] | food service | 52.1 | 60.6 | 55.0 | 58.1 |"
        }

## （8）IM-TQA（ACL2023）

**任务类型：** 中文隐式结构表格 QA和结构理解。

**核心特色和解决的问题：** 中文数据集，数据来自公开网页，覆盖多领域。**推理阶段不直接提供表头标注，模型需自主识别表头与数据单元格。** 主要用于真实场景中大量无标注的表格推理的情况，比如说日常生活中使用大模型进行表格理解，普通用户根本不可能去输入什么“TAB”“CELL”这种标记token，最多就是用空格和换行去间隔不同单元格。这就需要模型自己识别表格结构。除了表格QA，该数据集还可以用于表格结构理解任务。

**存在的问题：** 数值计算难度较低，数据集规模较小，只有中文没有英文，而且复杂表格情况较少。

        {
            "table_id": "2Hi9vmdU",
            "table_type": "vertical",
            "file_name": "垂直表格_31",
            "cell_ID_matrix": [
              [
                0,
                1,
                2
              ],
              [
                3,
                4,
                5
              ],
              [
                6,
                7,
                8
              ],
              [
                9,
                10,
                11
              ]
            ],
            "chinese_cell_value_list": [
              "项目",
              "本期金额/比例",
              "上期金额/比例",
              "研发支出金额",
              "4,172,343.06",
              "3,351,561.68",
              "研发支出占营业收入的比例",
              "1.56%",
              "3.7%",
              "研发支出中资本化的比例",
              "-",
              "-"
            ],
            "english_cell_value_list": [
              "project",
              "Current amount/proportion",
              "Amount/proportion of the previous period",
              "R&D expenditure amount",
              "4,172,343.06",
              "3,351,561.68",
              "Proportion of R&D expenditure in operating income",
              "1.56%",
              "3.7%",
              "Capitalized proportion of R&D expenditure",
              "-",
              "-"
            ],
            "column_attribute": [
              0,
              1,
              2
            ],
            "row_attribute": [],
            "column_index": [],
            "row_index": [
              3,
              6,
              9
            ]
          }


        {
    "table_id": "tdE2zGcU",
    "question_id": "tdE2zGcU_4",
    "file_name": "混杂表格_136",
    "chinese_question": "过滤器的过滤精度是多少？",
    "english_question": "What is the filter precision?",
    "answer_cell_list": [
      16
    ],
    "question_type": "single_cell"
  }
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
