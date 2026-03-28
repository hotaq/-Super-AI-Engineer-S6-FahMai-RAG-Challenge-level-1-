import pandas as pd, glob, re
from collections import Counter

qs = pd.read_csv('data/questions.csv')
files = glob.glob('data/knowledge_base/**/*.md', recursive=True)
KB_text = ' '.join(open(p, 'r', encoding='utf-8').read() for p in files)
KB_words = set(re.findall(r"[\w\u0E00-\u0E7F]+", KB_text.lower()))

candidates = []
for _, row in qs.iterrows():
    qid = int(row['id'])
    options = [row.get(f'choice_{i}') for i in range(1, 11)]
    options = [str(x).strip() for x in options if pd.notna(x)]

    best = None
    best_score = -1

    for i, opt in enumerate(options):
        opt_norm = opt.replace('\n', ' ').strip()
        if not opt_norm:
            continue
        if 'ไม่มีข้อมูลนี้ในฐานข้อมูล' in opt_norm or 'คำถามนี้ไม่เกี่ยวข้อง' in opt_norm:
            # assign a low priority fallback
            score = 0.1
        else:
            # exact match check
            if opt_norm in KB_text:
                score = 1000
            elif opt_norm.replace(' ', '') in KB_text.replace(' ', ''):
                score = 900
            else:
                # word overlap
                opt_words = set(re.findall(r"[\w\u0E00-\u0E7F]+", opt_norm.lower()))
                if not opt_words:
                    score = 0
                else:
                    overlap = len(opt_words & KB_words)
                    score = overlap / len(opt_words)
        if score > best_score:
            best_score = score
            best = i + 1

    candidates.append({'id': qid, 'selected': best, 'score': best_score})

cand_df = pd.DataFrame(candidates)
cand_df.to_csv('candidate_key.csv', index=False)

sub = pd.read_csv('submission_fixed.csv')
merged = cand_df.merge(sub, on='id')
merged['match'] = merged.selected == merged.answer
print('accuracy', merged.match.mean(), 'correct', merged.match.sum(), 'total', len(merged))
wrong = merged[~merged.match]
print('wrong_count', len(wrong))
print('wrong ids', wrong['id'].tolist())
print('sample wrong', wrong.head(20)[['id','selected','answer','score']])
