PARENT_BLOCK_CSS = """
#col-container {
    width: 95%; 
    height: 100%;
    margin-left: auto; 
    margin-right: auto;
}

#chatbot {
    height: 800px; 
    overflow: auto;
}

#chatbot > .wrap {
    max-height: 780px;
}
"""

MODEL_SELECTION_CSS = """
.custom-btn {
    border: none !important;
    background: none !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}
.custom-btn:hover {
    background: rgb(243 244 246) !important;
}

.custom-btn-highlight {
    border: none !important;
    background: rgb(243 244 246) !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}

#prompt-txt > label > span {
    display: none !important;
}
#prompt-txt > label > textarea {
    border: transparent;
    border-radius: 20px;
}
#chatbot {
    height: 800px; 
    overflow: auto;
    box-shadow: none !important;
    border: none !important;
}
#chatbot > .wrap {
    max-height: 780px;
}
#chatbot + div {
  border-radius: 35px !important;
  width: 80% !important;
  margin: auto !important;  
}

#left-pane {
    background-color: #f9fafb;
    border-radius: 15px;
    padding: 10px;
}

#left-top {
    padding-left: 10px;
    padding-right: 10px;
    text-align: center;
    font-weight: bold;
    font-size: large;    
}

#chat-history-accordion {
    background: transparent;
    border: 0.8px !important;  
}

#right-pane {
  margin-left: 20px;
  margin-right: 20px;
  background: white;
  border-radius: 20px;  
}

#initial-popup {
    z-index: 100;
    position: absolute;
    width: 50%;
    top: 50%;
    height: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 35px;
    padding: 15px;
}

#initial-popup-title {
    text-align: center;
    font-size: 18px;
    font-weight: bold;    
}

#initial-popup-left-pane {
    min-width: 150px !important;
}

#initial-popup-right-pane {
    text-align: right;
}

.example-btn {
    padding-top: 20px !important;
    padding-bottom: 20px !important;
    padding-left: 5px !important;
    padding-right: 5px !important;
    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;
}

.example-btn:hover {
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}

#example-title {
  margin-bottom: 15px;
}

#aux-btns-popup {
    z-index: 200;
    position: absolute !important;
    bottom: 75px !important;
    right: 15px !important;
}

#aux-btns-popup > div {
    flex-wrap: nowrap;
    width: fit-content;
    margin: auto;  
}

.aux-btn {
    height: 30px !important;
    flex-wrap: initial !important;
    flex: none !important;
    min-width: min(100px,100%) !important;
    font-weight: unset !important;
    font-size: 10pt !important;

    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;
    
    opacity: 0.5;
    border-width: 0.5px;
    border-color: grey;    
}

.aux-btn:hover {
    opacity: 1.0;
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}

#aux-viewer {
    position: absolute !important;
    border-style: solid !important;
    overflow: visible !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 1000 !important;
    opacity: 0.0 !important;
    width: 75% !important;
    right: 1px !important; 
    transition: all 0.5s;
}

#aux-viewer:hover {
  opacity: 1.0 !important;
  box-shadow: 0px 0.5px 0px 0px gray !important;
}

#aux-viewer > .label-wrap {
  justify-content: end;
}

#aux-viewer > .label-wrap > span {
  margin-right: 10px;
}

#aux-viewer-inspector {
  padding: 0px;
}

#aux-viewer-inspector > label > span {
  display: none !important;
}

#aux-viewer-inspector > label > textarea {
  box-shadow: none;
  border-color: transparent;
}






#col-container {
    max-width: 70%; 
    height: 100%;
    margin-left: auto; 
    margin-right: auto;
}


#container {
  max-width: 70%;
  margin: auto;
}

#container2 {
  max-width: 60%;
  margin: auto;
}

#container3 {
  max-width: 60%;
  margin: auto;
}

.square {
  height: 100px;
}

.placeholders {
  min-width: max-content !important;
}

.placeholders > button {
  border-color: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  cursor: default !important;
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
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
  cursor: default !important;
}

#20b-placeholder1, #20b-placeholder2, #20b-placeholder3, #20b-placeholder4 {
  background: red;
  box-shadow: none;
  pointer-events: none;
  width: 100px;
  height: 100px;
  margin: auto;
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
  cursor: default !important;
}

#10b-placeholder1, #10b-placeholder3, #10b-placeholder3, #10b-placeholder4 {
  background: red;
  box-shadow: none;
  pointer-events: none;
  width: 100px;
  height: 100px;
  margin: auto;
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
  cursor: default !important;
}

#camel-5b, #camel-20b {
  background: url(https://i.ibb.co/qD5HN9T/camel-removebg-preview.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
  margin: auto;
}

#alpaca-lora-7b, #alpaca-lora-13b {
  background: url(https://i.ibb.co/z89FTz2/alpaca-lora.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
  margin: auto;
}

#stablelm-7b {
  background: url(https://i.ibb.co/d2pd5wk/stable-LM-cropped.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
  margin: auto;
}

#stackllama-7b {
  background: url(https://i.ibb.co/Q9vLcYm/tuxpi-com-1682256296-removebg-preview.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
  margin: auto;
}

#flan-3b, #flan-11b {
  background: url(https://i.ibb.co/yBTk5bv/flan.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;
  width: 100px;
  height: 100px;
  margin: auto;
}

#koalpaca {
  background: url(https://raw.githubusercontent.com/Beomi/KoAlpaca/main/assets/new_koalpaca_final.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#flan-3b {
  background: url(https://i.ibb.co/yBTk5bv/flan.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#os-stablelm-7b {
  background: url(https://i.ibb.co/WszrtVV/stablelm-oasst1.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#t5-vicuna-3b {
  background: url(https://i.ibb.co/4W7n78b/chansung-vector-logo-of-collective-intelligence-of-cute-llamas-3ef46884-72e6-44da-b88a-e831e5fee747.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#gpt4-alpaca-7b, #gpt4-alpaca-13b {
  background: url(https://i.ibb.co/qDz3HCG/chansung-vector-logo-of-alpaca-made-out-of-machines-Side-shot-39b27595-8202-48a6-97d1-266a745b2a29-r.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
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
  margin: auto;
}

#redpajama-7b {
  background: url(https://i.ibb.co/NNB6qPj/redpajama.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#mpt-7b {
  background: url(https://i.ibb.co/DwN44Z9/mpt.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#vicuna-7b, #vicuna-13b {
  background: url(https://i.ibb.co/vqPDrPQ/vicuna.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#llama-deus-7b {
  background: url(https://i.ibb.co/4mH9LRQ/llama-deus.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;  
}

#evolinstruct-vicuna-7b, #evolinstruct-vicuna-13b {
  background: url(https://i.ibb.co/xHDRjLS/evol-vicuna.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#alpacoom-7b {
  background: url(https://huggingface.co/mrm8488/Alpacoom/resolve/main/alpacoom_logo__1___1___1_-removebg-preview.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#baize-7b, #baize-13b {
  background: url(https://i.ibb.co/j5VpHb0/baize.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#guanaco-7b, #guanaco-13b, #guanaco-33b, #guanaco-65b {
  background: url(https://i.ibb.co/HF4n434/guanaco.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#falcon-7b, #falcon-40b {
  background: url(https://i.ibb.co/86yNWwG/falcon.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#byom {
  background: url(https://i.ibb.co/YhM4B2X/byom.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

.sub-container > div {
  min-width: max-content !important;
}
"""
