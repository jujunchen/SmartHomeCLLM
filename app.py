import gradio as gr  
  
def echo(text):  
    return text  
  
iface = gr.Interface(fn=echo, inputs="text", outputs="text")  
iface.launch()