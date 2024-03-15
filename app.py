import gradio as gr  
import requests  
  
def process_input(user_input, user_input2):  
    try: 
        # 构造API请求  
        url = "http://183.129.211.90:22323/smartlife/chat"  
        headers = {"Content-Type": "application/json"}  # 根据你的API需求调整  
        data = {"messages": user_input}  # 构造请求体，这里假设API需要一个名为"param"的参数  
        
        # 发起请求并获取响应  
        response = requests.post(url, headers=headers, json=data)  
        
        # 处理响应并返回结果  
        if response.status_code == 200:  
            return response.json()
        else:  
            return {"error": "Failed to fetch data from API"} 
    except Exception as e:
        return {"error": str(e)}
  
# 创建Gradio界面  
iface = gr.Interface(  
    fn=process_input,  
    inputs=gr.Dropdown(["打开主卧灯", "关闭主卧灯"], label="选择指令"),
    outputs=gr.JSON(), 
    description= "做为演示，仅支持两个指令，后续我们将支持更多设备指令",
    title="绿城智能家居指令大模型Demo"  
)

iface.launch()  # 启动界面