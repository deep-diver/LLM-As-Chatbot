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

.template-txt {
    text-align: center;
    font-size: 15pt !important;
}

.message {
    margin: 0px !important;
}

.load-mode-selector:nth-child(3) {
    margin: auto !important;
    text-align: center !important;
    width: fit-content !important;
}

code {
    white-space: break-spaces !important;
}

.progress-view {
    background: transparent !important;
    border-radius: 25px !important;
}

#landing-container {
    width: 85%;
    margin: auto;
}

.landing-btn {
}

#landing-bottom {
    margin-top: 20px !important;
}

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

    @media (prefers-color-scheme: dark) {
      background-color: rgba(17,24,39,255) !important;
    }
}

#prompt-txt > label > span {
    display: none !important;
}
#prompt-txt > label > textarea {
    border: transparent;
    border-radius: 20px;
}
#chatbot {
    height: 800px !important; 
    overflow: auto;
    box-shadow: none !important;
    border: none !important;
}
#chatbot > .wrap {
    max-height: 780px !important;
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

  @media (prefers-color-scheme: dark) {
    background-color: rgba(31,41,55,255) !important;
  }  
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

  @media (prefers-color-scheme: dark) {
    background-color: rgba(31,41,55,255) !important;
  }
  
  @media screen and (max-width: 1000px) {
      margin: 0px !important;
  }
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

  @media (prefers-color-scheme: dark) {
    background: rgba(70,79,86,255) !important;
  }
}

.example-btn:hover {
  box-shadow: 0.3px 0.3px 0.3px gray !important;

  @media (prefers-color-scheme: dark) {
    background: rgba(34,37,42,255) !important;
  }
}

.example-btn:active {
  @media (prefers-color-scheme: dark) {
    background: rgba(70,79,86,255) !important;
  }
}

#example-title {
  margin-bottom: 15px;
}

#aux-btns-popup {
    z-index: 200;
    position: absolute !important;
    bottom: 75px !important;
    right: 40px !important;
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

    @media (prefers-color-scheme: dark) {
      opacity: 0.2 !important;
      color: black !important;
    }
}

.aux-btn:hover {
    opacity: 1.0;
    box-shadow: 0.3px 0.3px 0.3px gray !important;

    @media (prefers-color-scheme: dark) {
      opacity: 1.0 !important;
      box-shadow: 0.3px 0.3px 0.3px gray !important;
    }    
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

#global-context > label > span {
  display: none !important;
}

#chat-back-btn {
  background: transparent !important;
}

#chat-back-btn:hover {
  @media (prefers-color-scheme: dark) {
    background: rgb(75,85,99) !important;
  }
}

#chat-back-btn:active {
  @media (prefers-color-scheme: dark) {
    background: transparent !important;
  }
}

#col-container {
    max-width: 70%; 
    height: 100%;
    margin-left: auto; 
    margin-right: auto;
}


#model-selection-container {
  max-width: 80%;
  margin: auto;

  @media screen and (max-width: 1000px) {
      max-width: 100% !important;
  }
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

  @media (prefers-color-scheme: dark) {
    background-color: rgba(70,79,86,255) !important;
  }
}

.square:hover {
  @media (prefers-color-scheme: dark) {
    background-color: rgba(34,37,42,255) !important;
  }
}

.square:active {
  @media (prefers-color-scheme: dark) {
    background-color: rgba(70,79,86,255) !important;
  }
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
  background: url(https://i.ibb.co/hF9NL7r/koalpaca.png);
  background-repeat: no-repeat;
  background-size: 100px 100px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#kullm {
  background: url(https://i.ibb.co/6ZFqk4J/kullm.png);
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

#starchat-15b, #starchat-beta-15b { 
  background: url(https://i.ibb.co/QjPP0Vv/starcoder.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#redpajama-7b, #redpajama-instruct-7b {
  background: url(https://i.ibb.co/NNB6qPj/redpajama.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#mpt-7b, #mpt-30b {
  background: url(https://i.ibb.co/DwN44Z9/mpt.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#vicuna-7b, #vicuna-13b, #vicuna-7b-1-3, #vicuna-13b-1-3, #vicuna-33b-1-3 {
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
  background: url(https://i.ibb.co/DWWsZn7/guanaco.png);
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

#wizard-falcon-7b, #wizard-falcon-40b {
  background: url(https://i.ibb.co/415s0D4/wizard-falcon.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#nous-hermes-13b, #nous-hermes-13b-llama2, #nous-hermes-7b-llama2, #nous-hermes-70b {
  background: url(https://i.ibb.co/sm8VgtL/nous-hermes.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#nous-puffin-13b-llama2 {
  background: url(https://i.ibb.co/L1RDS9q/puffin.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#airoboros-7b, #airoboros-13b {
  background: url(https://i.ibb.co/NLchBkB/airoboros.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#samantha-7b, #samantha-13b, #samantha-33b, #samantha-70b {
  background: url(https://i.ibb.co/72t5pyP/samantha.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#lazarus-30b {
  background: url(https://i.ibb.co/Zm2Bdzt/lazarus.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#chronos-13b, #chronos-33b {
  background: url(https://i.ibb.co/sQZ3L8j/chronos.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#wizardlm-13b, #wizardlm-30b, #wizardlm-13b-1-2, #wizardlm-70b {
  background: url(https://i.ibb.co/SRXWKz9/WizardLM.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#wizard-vicuna-13b, #wizard-vicuna-30b {
  background: url(https://i.ibb.co/MDTbWfz/wizard-vicuna-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#wizard-coder-15b {
  background: url(https://i.ibb.co/NC883qm/wizard-coder-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#openllama-7b, #openllama-13b {
  background: url(https://i.ibb.co/Wsq1SQ8/openllama-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#orcamini-7b, #orcamini-13b, #orcamini-70b {
  background: url(https://i.ibb.co/fMMD92f/orca-mini-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#xgen-7b {
  background: url(https://i.ibb.co/qFRbJGD/xgen-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#llama2-7b, #llama2-13b {
  background: url(https://i.ibb.co/dDr1Kv7/llama2-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#upstage-llama-30b, #upstage-llama2-70b, #upstage-llama2-70b-2 {
  background: url(https://i.ibb.co/FX3Vf9K/upstage.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#platypus2-70b {
  background: url(https://i.ibb.co/wpv3Q43/platypus2.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#stable-beluga2-70b {
  background: url(https://i.ibb.co/m0jd3P3/freewilly-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#godzilla-70b {
  background: url(https://i.ibb.co/XV6C2Dm/godzilla-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#codellama-7b, #codellama-13b, #codellama-34b {
  background: url(https://i.ibb.co/RCv6nLV/code-llama-mid.png);
  background-repeat: no-repeat;
  background-size: 100px 95px;
  color: transparent;  
  width: 100px;
  height: 100px;
  margin: auto;
}

#replit-3b {
  background: url(https://i.ibb.co/BrKCKYq/replit.png);
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

#chosen-model {
  background: url(https://i.ibb.co/dLmNh2v/chosen.png);
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
