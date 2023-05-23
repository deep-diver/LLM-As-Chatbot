GET_LOCAL_STORAGE = """
function() {
  globalThis.setStorage = (key, value)=>{
    localStorage.setItem(key, JSON.stringify(value));
  }
  globalThis.getStorage = (key, value)=>{
    return JSON.parse(localStorage.getItem(key));
  }

  var local_data = getStorage('local_data');
  var history = [];

  if(local_data) {
    local_data[0].pingpongs.forEach(element =>{ 
      history.push([element.ping, element.pong]);
    });
  }
  else {
    local_data = [];
    for (let step = 0; step < 10; step++) {
      local_data.push({'ctx': '', 'pingpongs':[]});
    }
    setStorage('local_data', local_data);
  }

  if(history.length == 0) {
    document.querySelector("#initial-popup").classList.remove('hide');
  }
  
  return [history, local_data];
}
"""

UPDATE_LEFT_BTNS_STATE = """
(v)=>{
  document.querySelector('.custom-btn-highlight').classList.add('custom-btn');
  document.querySelector('.custom-btn-highlight').classList.remove('custom-btn-highlight');

  const elements = document.querySelectorAll(".custom-btn");

  for(var i=0; i < elements.length; i++) {
    const element = elements[i];
    if(element.textContent == v) {
      console.log(v);
      element.classList.add('custom-btn-highlight');
      element.classList.remove('custom-btn');
      break;
    }
  }
}""" 