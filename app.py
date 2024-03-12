import gradio as gr  
import requests  
  
def process_input(user_input):  
    # try: 
    #     # 构造API请求  
    #     url = "https://smarthome-dev.gtdreamlife.com/smartlife/chat"  
    #     headers = {"Content-Type": "application/json"}  # 根据你的API需求调整  
    #     data = {"messages": user_input}  # 构造请求体，这里假设API需要一个名为"param"的参数  
        
    #     # 发起请求并获取响应  
    #     response = requests.post(url, headers=headers, json=data)  
        
    #     # 处理响应并返回结果  
    #     if response.status_code == 200:  
    #         return response.json()
    #     else:  
    #         return {"error": "Failed to fetch data from API"} 
    # except Exception as e:
    #     print(e)
    #     return {"error": str(e)}
    return {"response": "Failed to fetch data from API"}
  
# 创建Gradio界面  
iface = gr.Interface(  
    fn=process_input,  
    inputs=gr.Textbox(label="输入指令222"),  
    outputs=gr.JSON(), 
    title="智能家居指令大模型Demo"  
)

# 启动界面  
iface.launch()