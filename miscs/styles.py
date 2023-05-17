PARENT_BLOCK_CSS = """
#col-container {
    width: 95%; 
    height: 100%;
    margin-left: auto; 
    margin-right: auto;
}

#chatbot {
    height: 520px; 
    overflow: auto;
}
"""

MODEL_SELECTION_CSS = """
#col-container {
    width: 100%; 
    height: 100%;
    margin-left: auto; 
    margin-right: auto;
}

#chatbot {
    height: 520px; 
    overflow: auto;
}

#container {
  width: 600px;
  margin: auto;
}

#container2 {
  width: 600px;
  margin: auto;  
}

.square {
  height: 100px;
}

.center {
  text-align: center;
  overflow: hidden;
}

#30b-placeholder1, #30b-placeholder2, #30b-placeholder3, #30b-placeholder4 {
  background: red;
  box-shadow: none;
  pointer-events: none;
  width: 100px;
  height: 100px;
}

#20b-placeholder1, #20b-placeholder2, #20b-placeholder3, #20b-placeholder4 {
  background: red;
  box-shadow: none;
  pointer-events: none;
  width: 100px;
  height: 100px;
}

#10b-placeholder1, #10b-placeholder3, #10b-placeholder3, #10b-placeholder4 {
  background: red;
  box-shadow: none;
  pointer-events: none;
  width: 100px;
  height: 100px;  
}

#camel-5b, #camel-20b {
  background: url(https://i.ibb.co/qD5HN9T/camel-removebg-preview.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
}

#alpaca-lora-7b, #alpaca-lora-13b {
  background: url(https://i.ibb.co/z89FTz2/alpaca-lora.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
}

#stablelm-7b {
  background: url(https://i.ibb.co/d2pd5wk/stable-LM-cropped.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
}

#stackllama-7b {
  background: url(https://i.ibb.co/Q9vLcYm/tuxpi-com-1682256296-removebg-preview.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
}

#flan-3b, #flan-11b {
  background: url(https://i.ibb.co/yBTk5bv/flan.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
}

#koalpaca {
  background: url(https://raw.githubusercontent.com/Beomi/KoAlpaca/main/assets/new_koalpaca_final.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#flan-3b {
  background: url(https://i.ibb.co/yBTk5bv/flan.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#os-stablelm-7b {
  background: url(https://i.ibb.co/WszrtVV/stablelm-oasst1.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#t5-vicuna-3b {
  background: url(https://i.ibb.co/4W7n78b/chansung-vector-logo-of-collective-intelligence-of-cute-llamas-3ef46884-72e6-44da-b88a-e831e5fee747.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#gpt4-alpaca-7b, #gpt4-alpaca-13b {
  background: url(https://i.ibb.co/qDz3HCG/chansung-vector-logo-of-alpaca-made-out-of-machines-Side-shot-39b27595-8202-48a6-97d1-266a745b2a29-r.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#stable-vicuna-13b {
  background: url(https://i.ibb.co/b6Vv6Jh/sv.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;  
}

#starchat-15b { 
  background: url(https://i.ibb.co/QjPP0Vv/starcoder.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;    
}

#redpajama-7b {
  background: url(https://i.ibb.co/NNB6qPj/redpajama.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#mpt-7b {
  background: url(https://i.ibb.co/DwN44Z9/mpt.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#vicuna-7b, #vicuna-13b {
  background: url(https://i.ibb.co/vqPDrPQ/vicuna.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;  
}

#llama-deus-7b {
  background: url(https://i.ibb.co/4mH9LRQ/llama-deus.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

#evolinstruct-vicuna-7b, #evolinstruct-vicuna-13b {
  background: url(https://i.ibb.co/xHDRjLS/evol-vicuna.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
}

"""
