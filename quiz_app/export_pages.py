"""Export the curated MCQ bank as a GitHub Pages data file."""
import json
from pathlib import Path
from quiz_app.catalog import BANK

root = Path(__file__).resolve().parents[1]
questions = []
for q in BANK:
    m = q['mcq']
    questions.append({
        'id': q['id'], 'field': q['field'], 'level': q['level'],
        'title': q['title'], 'prompt': m['prompt'], 'template': m['template'],
        'options': m['options'], 'answer': m['answer'],
        'explanation': q['explanation'],
        'source': f"{q['source']['path']} · {q['source']['cell']}번째 셀",
    })
(root / 'docs' / 'questions.js').write_text(
    'window.QUESTIONS=' + json.dumps(questions, ensure_ascii=False, separators=(',', ':')) + ';\n',
    encoding='utf-8',
)
print(f'Exported {len(questions)} questions')
