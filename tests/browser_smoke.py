"""Optional: run against ./run_quiz.sh with Playwright + Chromium installed."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quiz_app.catalog import BANK, BY_ID
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765')
    expect(page.locator('#bank-count')).to_have_text(str(len(BANK)))
    expect(page.locator('#kind')).to_have_value('mcq')
    page.screenshot(path='/tmp/ais-home.png',full_page=True)
    for field in ['LLM','Vision','Data','RAG']:
        page.locator(f'[data-field="{field}"]').click()
    page.select_option('#kind','code');page.select_option('#level','hard');page.fill('#count','2')
    page.click('#start');expect(page.locator('#quiz')).to_be_visible()
    qid=page.evaluate('session.questions[index].id')
    page.fill('#editor',BY_ID[qid]['answer'])
    expect(page.locator('#save-state')).to_have_text('저장됨')
    page.reload();expect(page.locator('#resume')).to_be_visible();page.click('#resume')
    expect(page.locator('#editor')).to_have_value(BY_ID[qid]['answer'])
    page.click('#submit');expect(page.locator('#feedback')).to_contain_text('정답입니다',timeout=30000)
    page.screenshot(path='/tmp/ais-quiz.png',full_page=True)
    with page.expect_download() as download:
        page.click('#download')
    notebook=json.loads(Path(download.value.path()).read_text())
    assert notebook['nbformat']==4
    compile(notebook['cells'][2]['source'][0],'<export>','exec')
    page.click('#next');page.fill('#editor','pass');page.click('#submit')
    expect(page.locator('#feedback')).to_contain_text('다시 확인',timeout=30000)
    page.click('#finish');page.click('#confirm-finish')
    expect(page.locator('#results')).to_be_visible(timeout=30000)
    page.click('#retry-wrong');expect(page.locator('#quiz')).to_be_visible()
    assert page.evaluate('session.questions.length')==1
    page.click('#home-nav')
    page.locator('[data-mode="exam"]').click();page.select_option('#kind','mcq');page.click('#start')
    expect(page.locator('#quiz')).to_be_visible()
    assert page.evaluate('session.questions.length')==20
    expect(page.locator('#hint')).to_be_hidden()
    page.locator('#options input').first.check();page.click('#submit')
    expect(page.locator('#feedback')).to_have_text('')
    # Timer reaching zero must finish automatically without showing the dialog.
    page.evaluate('session.deadline = Date.now()/1000 - 1')
    expect(page.locator('#results')).to_be_visible(timeout=30000)
    page.click('#review-nav');expect(page.locator('#history-items details').first).to_be_visible()
    page.screenshot(path='/tmp/ais-history.png',full_page=True)
    page.click('#home-nav');page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile horizontal overflow'
    page.screenshot(path='/tmp/ais-mobile.png',full_page=True)
    assert not errors,errors
    print('Browser checks passed: code, save/reload, grading, partial flow, notebook export, review, exam timer, history, mobile.')
    browser.close()
