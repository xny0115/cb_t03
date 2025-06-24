import pytest
import json
import subprocess

pytestmark = [pytest.mark.requires_node, pytest.mark.requires_torch]


def send_two_messages():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
new window.MutationObserver(() => { box.scrollTop = 0; }).observe(box, { childList: true });
function appendChat(role, text){
  const wrap = window.document.createElement('div');
  wrap.className = role === 'USER' ? 'chat-msg user' : 'chat-msg bot';
  const bubble = window.document.createElement('div');
  bubble.className = role === 'USER' ? 'user-msg' : 'bot-msg';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  box.prepend(wrap);
}
for(let i=0;i<30;i++) appendChat('USER','x'+i);
console.log(JSON.stringify({ top: box.scrollTop, height: box.scrollHeight, client: box.clientHeight }));
"""
    out = subprocess.check_output(['node', '-e', script])
    return json.loads(out.decode())


def test_auto_scroll():
    res = send_two_messages()
    assert res['top'] == 0
