import pytest
import json, subprocess

pytestmark = [pytest.mark.requires_node, pytest.mark.requires_torch]


def run_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
if(!window.requestAnimationFrame){ window.requestAnimationFrame = cb => setTimeout(cb,16); }
function appendChat(role,text,latency=0){
  const wrap = window.document.createElement('div');
  wrap.className = role==='USER'?'chat-msg user':'chat-msg bot';
  const bubble = window.document.createElement('div');
  bubble.className = role==='USER'?'user-msg':'bot-msg';
  bubble.textContent = text;
  const meta = window.document.createElement('div');
  meta.className='meta';
  meta.textContent=new Date().toLocaleTimeString();
  if(role!=='USER') meta.textContent += ' · '+ latency+' ms';
  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  box.prepend(wrap);
  function scrollTop(){box.scrollTop=0;}
  scrollTop();
  window.requestAnimationFrame(scrollTop);
  window.setTimeout(scrollTop,30);
}
appendChat('USER','x');
setTimeout(()=>{console.log(JSON.stringify({top:box.scrollTop,height:box.scrollHeight,client:box.clientHeight}));},40);
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_scroll_multistep():
    res = run_script()
    assert res["top"] == 0
