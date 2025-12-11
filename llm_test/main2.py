from openai import OpenAI
import json

with open(r"D:\论文\eval_data_initial\in_domain_test\fetaqa_test.json","r",encoding="utf-8") as f:
    d = json.load(f)


client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="?????"
)

right = 0
try:
    for i in range(len(d)):
        content = "instruction:"+d[i]["instruction"]+" "+"input_seg:"+d[i]["input_seg"]+"question:"+d[i]["question"]+" "
        content1="请根据这些信息，自行解读表格结构，用英文回答问题。仅需要输出最终答案，不要保留推断内容等中间过程。但是回答尽量完整丰富。"
        content+=content1
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            print(reasoning)
        res = completion.choices[0].message.content
        c = "instruction:"+d[i]["instruction"]+" "+"input_seg:"+d[i]["input_seg"]+"question:"+d[i]["question"]+" "
        c+="根据以上信息，判断下面的回答是否正确。你认为正确输出1，错误输出0。仅可以输出1或者0，且不要有多余输出，禁止输出任何中间推导过程。"
        c+="answer:"+res
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.2-Exp",
            messages=[{"role": "user", "content": c}],
            temperature=0.01,
            top_p=0.9,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            print(reasoning)
        res1 = completion.choices[0].message.content
        if res1=="1":
            right+=1
        print(res1)
except:
    print("tot:",end = str(len(d)))
    print("right:",end = str(right))
    print(right/len(d))
finally:
    print("tot:",end = str(len(d)))
    print("right:",end = str(right))

    print(right/len(d))
