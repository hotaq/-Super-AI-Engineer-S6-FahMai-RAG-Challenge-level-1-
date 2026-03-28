import pandas as pd, glob, re

# Load dataset
qs = pd.read_csv('data/questions.csv')
files = glob.glob('data/knowledge_base/**/*.md', recursive=True)
KB = {p: open(p, 'r', encoding='utf-8').read() for p in files}

# no-data options patterns
no_data_patterns = ['ไม่มีข้อมูลนี้ในฐานข้อมูล', 'คำถามนี้ไม่เกี่ยวข้อง']

final_rows = []
for _, row in qs.iterrows():
    qid = int(row['id'])
    question = str(row['question'])
    options = [row.get(f'choice_{i}') for i in range(1, 11)]
    options = [(i+1, str(opt).strip()) for i,opt in enumerate(options) if pd.notna(opt)]

    # manual overrides from verified conversation
    manual = {
        7: 1,
        10: 7,
        94: 4,
        95: 1,
        98: 5,
    }
    if qid in manual:
        final_rows.append({'id': qid, 'answer': manual[qid], 'reason': 'manual override'})
        continue

    best_choice = None
    best_score = -1
    for choice_num, opt in options:
        if not opt:
            continue

        # skip no-data for now
        is_no_data = any(p in opt for p in no_data_patterns)

        # exact count in KB text
        exact_count = sum(1 for t in KB.values() if opt in t)

        # substring match of words
        opt_words = set(re.findall(r"[\w\u0E00-\u0E7F]+", opt.lower()))
        q_words = set(re.findall(r"[\w\u0E00-\u0E7F]+", question.lower()))
        overlap_words = opt_words & q_words
        overlap_score = len(overlap_words) / max(len(opt_words), 1)

        # semantic clue: product names or specs in question
        product_names = ['AirBook 14', 'X9 Pro', 'StormBook G5', 'S3', 'S3 Pro', 'S3 Ultra', 'HeadPro', 'HeadOn', 'NovaBuds', 'SoundBar']
        prod_score = 0
        for pn in product_names:
            if pn.lower() in question.lower() and pn.lower() in opt.lower():
                prod_score = 0.5
                break

        score = exact_count * 100 + overlap_score * 10 + prod_score
        if is_no_data:
            score *= 0.01

        if score > best_score:
            best_score = score
            best_choice = choice_num

    if best_choice is None:
        best_choice = options[0][0] if options else 1

    final_rows.append({'id': qid, 'answer': best_choice, 'reason': f'auto score {best_score:.2f}'})

final_df = pd.DataFrame(final_rows).sort_values('id')
final_df.to_csv('submission_final.csv', index=False)

# Print summary
print('wrote submission_final.csv')
print(final_df.head(15))

# compare against submission_fixed for still existing mismatches
subf = pd.read_csv('submission_fixed.csv')
merged = final_df.merge(subf, on='id', suffixes=('_final', '_fixed'))
merged['match'] = merged.answer_final == merged.answer_fixed
print('match fraction with submission_fixed', merged.match.mean())
print('not match', merged[~merged.match].shape[0])
print(merged[~merged.match].head(20))
