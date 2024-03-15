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
    article= """1、为了部署到openXLab，我们开发了跟演示视频不同的Demo，该Demo也对接了实验室的IOT平台，能真实控制实验室设备，该Demo仅支持两个指令。
    <br/>
    2、直播地址能够实时看到设备控制情况，<b>由于摄像头问题，请在手机端打开观看，视频流有20秒左右延迟，请评委耐心等待一下...有时候会打不开，麻烦评委多尝试一下...
    <br/>直播地址：http://test.aliali.vip/video</b>
    """,
    title="绿城智能家居指令大模型Demo"  
)

iface.launch()  # 启动界面