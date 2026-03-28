import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUESTIONS_CSV = ROOT / "data" / "questions.csv"
SUBMISSION_CSV = ROOT / "submission.csv"
OUTPUT_DIR = ROOT / "artifacts" / "training"
OUTPUT_CSV = OUTPUT_DIR / "imitation_train.csv"
OUTPUT_JSONL = OUTPUT_DIR / "imitation_train.jsonl"


SYSTEM_PROMPT = """คุณเป็นระบบตอบคำถามแบบหลายตัวเลือกของร้านฟ้าใหม่ (FahMai)

ตอบจากฐานข้อมูลของร้านเท่านั้น และตอบเป็นรูปแบบ:
ANSWER: X

โดย X เป็นตัวเลข 1-10 เพียงค่าเดียว
"""


def load_questions() -> dict[int, dict[str, str]]:
    with QUESTIONS_CSV.open(newline="", encoding="utf-8") as handle:
        return {int(row["id"]): row for row in csv.DictReader(handle)}


def load_submission() -> dict[int, int]:
    with SUBMISSION_CSV.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["id"]): int(str(row["answer"]).strip())
            for row in csv.DictReader(handle)
        }


def build_user_prompt(question_row: dict[str, str]) -> str:
    lines = [f"คำถาม: {question_row['question']}", "", "ตัวเลือก:"]
    for idx in range(1, 11):
        lines.append(f"{idx}. {question_row[f'choice_{idx}']}")
    lines.append("")
    lines.append("ตอบเป็น ANSWER: X เท่านั้น")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    submission = load_submission()

    csv_rows = []
    jsonl_rows = []

    for qid in sorted(questions):
        row = questions[qid]
        answer = submission[qid]
        answer_text = row[f"choice_{answer}"]
        user_prompt = build_user_prompt(row)

        csv_row = {
            "id": qid,
            "question": row["question"],
            "answer": answer,
            "answer_text": answer_text,
        }
        for idx in range(1, 11):
            csv_row[f"choice_{idx}"] = row[f"choice_{idx}"]
        csv_rows.append(csv_row)

        jsonl_rows.append(
            {
                "id": qid,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": f"ANSWER: {answer}"},
                ],
                "answer": answer,
                "answer_text": answer_text,
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "question"] + [f"choice_{idx}" for idx in range(1, 11)] + ["answer", "answer_text"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source_submission": SUBMISSION_CSV.name,
                "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
                "output_jsonl": str(OUTPUT_JSONL.relative_to(ROOT)),
                "rows": len(csv_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
