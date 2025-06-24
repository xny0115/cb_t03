import pytest
import json, subprocess

pytestmark = [pytest.mark.requires_node, pytest.mark.requires_torch]


def run_dom():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory"></div>', { runScripts: 'outside-only' });
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
appendChat('BOT','hi',23);
console.log(JSON.stringify({html:box.innerHTML}));
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_timestamp_latency():
    res = run_dom()
    assert "ms" in res["html"]
