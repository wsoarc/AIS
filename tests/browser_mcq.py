"""Optional browser regression for compact MCQ blanks and notebook export."""
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quiz_app.catalog import BANK, BY_ID
from playwright.sync_api import sync_playwright,expect

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765')
    expect(page.locator('#bank-count')).to_have_text(str(len(BANK)))
    expect(page.locator('#kind')).to_have_value('mcq')
    for qid in ['core-llm-attention','core-llm-lora-init','core-od-snapkv','core-rag-mcp-discovery','core-data-lstm']:
        page.select_option('#kind','mcq')
        page.evaluate('(id) => start([id])',qid)
        expect(page.locator('#quiz')).to_be_visible()
        public=page.evaluate('session.questions[0]')
        assert public['template']==BY_ID[qid]['mcq']['template']
        assert public['template'].count('### 작성 필요 ###')==1
        assert all(len(o['code'].splitlines())<=3 for o in public['options'])
        correct=next(i for i,o in enumerate(public['options']) if o['code']==BY_ID[qid]['mcq']['answer'])
        if qid=='core-llm-attention':page.screenshot(path='/tmp/ais-mcq-blank.png',full_page=True)
        page.locator('#options input').nth(correct).check()
        page.click('#submit')
        expect(page.locator('#feedback')).to_contain_text('정답입니다')
        expect(page.locator('#feedback .code')).to_have_text(BY_ID[qid]['mcq']['answer'])
        with page.expect_download() as download:page.click('#download')
        nb=json.loads(Path(download.value.path()).read_text())
        src=nb['cells'][2]['source'][0]
        compile(src,'<mcq notebook>','exec')
        assert '### 작성 필요 ###' not in src
        if qid=='core-data-lstm':
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.click('#home-nav')
    page.select_option('#kind','code')
    page.evaluate("start(['core-llm-attention'])")
    expect(page.locator('#editor')).to_be_visible()
    assert page.locator('#template').inner_text()==BY_ID['core-llm-attention']['template']
    page.click('#home-nav')
    page.select_option('#kind','code')
    page.evaluate("start(['core-od-smooth'])")
    expect(page.locator('#editor')).to_be_visible()
    page.fill('#editor',BY_ID['core-od-smooth']['answer'])
    page.click('#submit')
    expect(page.locator('#feedback')).to_contain_text('정답입니다',timeout=30000)
    assert not errors,errors
    print('Compact MCQ display, blank-only feedback, notebook export, mobile layout, and original code template passed.')
    browser.close()
