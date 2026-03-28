import pandas as pd, glob, re

qs = pd.read_csv('data/questions.csv')
files = glob.glob('data/knowledge_base/**/*.md', recursive=True)
KB = {p: open(p, 'r', encoding='utf-8').read() for p in files}

results = []
for _, row in qs.iterrows():
    qid = int(row['id'])
    options = [row.get(f'choice_{i}') for i in range(1, 11)]
    options = [str(x) for x in options if pd.notna(x)]
    selected = None

    for i, opt in enumerate(options):
        if not opt or 'ไม่มีข้อมูลนี้' in opt or 'ไม่เกี่ยวข้อง' in opt:
            continue
        for t in KB.values():
            if opt in t:
                selected = i + 1
                break
        if selected is not None:
            break

    if selected is None:
        for i, opt in enumerate(options):
            if 'ไม่มีข้อมูลนี้ในฐานข้อมูล' in opt or 'คำถามนี้ไม่เกี่ยวข้อง' in opt:
                selected = i + 1
                break

    if selected is None and options:
        selected = 1

    results.append({'id': qid, 'selected': selected})

cand = pd.DataFrame(results)
sub = pd.read_csv('submission_fixed.csv')
merged = cand.merge(sub, on='id')
merged['match'] = merged.selected == merged.answer

print('total', len(merged))
print('correct_match', merged.match.sum(), 'accuracy', merged.match.mean())
wrong = merged[~merged.match]
print('wrong_count', len(wrong))
print('wrong ids', wrong['id'].tolist())
print('sample wrong')
print(wrong.head(20))

cand.to_csv('auto_selected.csv', index=False)
print('auto_selected.csv written')
