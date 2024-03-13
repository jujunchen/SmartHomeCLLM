import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 修改为模型路径
model_name_or_path = "/app/models/internlm2-chat-7b-ft/Greentown_SmartHomeCLLM/"


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """
你现在是一个智能家居AI助手, 能够从用户上下文提取出用户的动作(action), 设备(device), 空间(space), 接口(api), 回复(reponse), 场景(scene), 设备id(device_id), 场景id(scene_id), 没有的字段返回空;
设备数据从这里匹配: (device_id=1,客厅,射灯);(device_id=4,客厅,筒灯);(device_id=13,客厅,灯带);(device_id=11,客厅,窗帘);
场景数据从这里匹配: (scene_id=1,回家);(scene_id=2,离家);(scene_id=4,洗浴);(scene_id=5,睡眠); 
"""

messages = [(system_prompt, '')]

def process_input(input_text):
    response, history = model.chat(tokenizer, input_text, history=messages)
    return response

# 创建 gradio 接口  
iface = gr.Interface(  
    fn=process_input,  
    inputs=gr.Textbox(label="输入指令"),  
    outputs=gr.JSON(), 
    title="绿城智能家居指令大模型Demo"  
)  
iface.launch()