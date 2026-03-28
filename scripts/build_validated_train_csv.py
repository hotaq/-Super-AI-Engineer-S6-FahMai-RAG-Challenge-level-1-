import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUESTIONS_CSV = ROOT / "data" / "questions.csv"
SUBMISSION_FIXED_CSV = ROOT / "submission_fixed.csv"
OUTPUT_DIR = ROOT / "artifacts" / "validated"

# Explicitly reviewed against the local knowledge base.
MANUAL_VALIDATIONS = {
    31: {
        "answer": 4,
        "note": "Mega Sale return window is 7 days, not the normal policy window.",
    },
    54: {
        "answer": 9,
        "note": "StormPhone X9 Pro Max page does not state country of manufacture.",
    },
    55: {
        "answer": 9,
        "note": "Store info has company details but no annual revenue figure.",
    },
    56: {
        "answer": 9,
        "note": "Knowledge base has AirBook 14 and 15, but no AirBook 13 entry.",
    },
    57: {
        "answer": 9,
        "note": "X9 Pro page does not include screen-to-body ratio.",
    },
    58: {
        "answer": 9,
        "note": "Buds Z5 Pro page has no average user review score.",
    },
    59: {
        "answer": 9,
        "note": "X9 Pro Max page does not list SAR value.",
    },
    60: {
        "answer": 10,
        "note": "Public holidays in 2569 are unrelated to FahMai.",
    },
    61: {
        "answer": 10,
        "note": "Bangkok-Chiang Mai airfare is unrelated to FahMai.",
    },
    62: {
        "answer": 10,
        "note": "Savings-account interest rate is unrelated to FahMai.",
    },
    63: {
        "answer": 10,
        "note": "Pad kra pao recipe is unrelated to FahMai.",
    },
    22: {
        "answer": 6,
        "note": "Screen damage is not covered by normal warranty; Care+ covers up to 2 repairs per year with 20% co-pay.",
    },
    9: {
        "answer": 4,
        "note": "Rugged R1 has IP69K and MIL-STD-810H, but that does not make it suitable for deep diving, and water damage is not warranty-covered.",
    },
    14: {
        "answer": 4,
        "note": "X9 Pro includes the 67W charger and USB-C cable in the box; the separate charger SKU does not include a cable.",
    },
    41: {
        "answer": 7,
        "note": "FlexBook Detach does not include a keyboard; the keyboard is sold separately or in the DN-LT-018 bundle.",
    },
    47: {
        "answer": 2,
        "note": "DaoNuea's 27-inch 4K product is All-in-One 27 at 34990 THB; ArcWave ProView 27 is not DaoNuea.",
    },
    49: {
        "answer": 1,
        "note": "The standard HeadOn 300 SKU lists Black, White, and Navy Blue. FahMai Blue belongs to the separate FahMai Edition SKU.",
    },
    65: {
        "answer": 3,
        "note": "X9 and X9 FE share the S9 chip; FE is cheaper because it loses OIS, loses ultrawide, and uses Polycarbonate instead of aluminum.",
    },
    66: {
        "answer": 7,
        "note": "NovaTech partner products have no on-site service; SlimBook 14 must be sent to a NovaTech service center.",
    },
    76: {
        "answer": 5,
        "note": "Both G5 variants use SO-DIMM RAM; current G5 is DDR5 and 2024 is DDR4.",
    },
    82: {
        "answer": 6,
        "note": "32990 + 12990 + 1890 = 47870.",
    },
    83: {
        "answer": 4,
        "note": "Gold: floor(32990/100)=329, then 329*1.5=493.5, round down to 493 points.",
    },
    84: {
        "answer": 5,
        "note": "8000 points = 4000 THB discount; 20% cap is 7998 THB, so 4000 THB is the max usable.",
    },
    85: {
        "answer": 4,
        "note": "Over-ear models under 5000 THB are HeadOn 300, HeadOn 300 FahMai Edition, GameStorm H1, and HeadOn 500.",
    },
    88: {
        "answer": 3,
        "note": "Only the two AirBook 14 variants satisfy fanless, <=1.2kg, and >15h battery life.",
    },
    89: {
        "answer": 4,
        "note": "Only Watch S3 Pro fits ECG + NFC Pay + swim + <=10000 THB.",
    },
    90: {
        "answer": 5,
        "note": "Only HeadPro X1 has ANC + LDAC + at least 30h battery within the 13000 THB budget.",
    },
    32: {
        "answer": 2,
        "note": "Orders in the 'paid / preparing to ship' state can still be canceled through the app or website unless packing and tracking creation have already completed.",
    },
    92: {
        "answer": 4,
        "note": "Among wireless earbuds/headphones at or below 8000 THB, only Buds Z5 Pro has both ANC and LDAC.",
    },
    97: {
        "answer": 7,
        "note": "ArcWave warranty excludes accidental damage, and partner brands cannot buy FahMai Care+.",
    },
    98: {
        "answer": 5,
        "note": "Standard shipping is free over 500 THB; SoundBar Pro 500 adds 200 THB heavy-item fee plus 100 THB each for floors 4, 5, and 6 with no elevator.",
    },
    78: {
        "answer": 3,
        "note": "StormBook G7 has on-site service only in year 1, while Mini PC M1, as a DaoNuea desktop, has on-site service for the full 3-year warranty.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_submission_answers(path: Path) -> dict[int, int]:
    rows = read_csv(path)
    answers: dict[int, int] = {}
    for row in rows:
        qid = int(str(row["id"]).strip())
        answer = int(str(row["answer"]).strip())
        answers[qid] = answer
    return answers


def collect_submission_runs() -> dict[int, dict[str, int]]:
    patterns = [
        "submission.csv",
        "submission_2.csv",
        "submission_3.csv",
        "submission_final.csv",
        "submission_fixed.csv",
        "submission_source_aware.csv",
        "submissions/fixed_512_128.csv",
        "archive/submissions/submission_2.csv",
        "archive/submissions/submission_3.csv",
        "archive/submissions/submission_final.csv",
        "archive/submissions/submission_source_aware.csv",
    ]
    collected: dict[int, dict[str, int]] = defaultdict(dict)
    for rel_path in patterns:
        path = ROOT / rel_path
        if not path.exists():
            continue
        for qid, answer in load_submission_answers(path).items():
            collected[qid][path.name] = answer
    return collected


def build_outputs() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = read_csv(QUESTIONS_CSV)
    fixed_answers = load_submission_answers(SUBMISSION_FIXED_CSV)
    submission_runs = collect_submission_runs()

    cleaned_rows: list[dict[str, int]] = []
    full_rows: list[dict[str, object]] = []
    validated_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for row in questions:
        qid = int(row["id"])
        current_answer = fixed_answers[qid]
        votes = submission_runs.get(qid, {})
        vote_counter = Counter(votes.values())
        majority_answer = None
        majority_count = 0
        if vote_counter:
            majority_answer, majority_count = vote_counter.most_common(1)[0]
        num_runs = len(votes)
        consensus_ratio = (majority_count / num_runs) if num_runs else 0.0

        if qid in MANUAL_VALIDATIONS:
            final_answer = MANUAL_VALIDATIONS[qid]["answer"]
            label_source = "manual_kb_review"
            confidence = 1.0
            note = MANUAL_VALIDATIONS[qid]["note"]
        else:
            final_answer = current_answer
            label_source = "submission_fixed"
            confidence = consensus_ratio if current_answer == majority_answer else 0.0
            if majority_answer is None:
                note = "No comparison submissions available."
            elif current_answer == majority_answer:
                note = f"Matches majority vote {majority_answer} from {majority_count}/{num_runs} runs."
            else:
                note = (
                    f"Current answer {current_answer} disagrees with majority vote "
                    f"{majority_answer} from {majority_count}/{num_runs} runs."
                )

        cleaned_rows.append({"id": qid, "answer": final_answer})

        full_row = {
            "id": qid,
            "question": row["question"],
            "answer": final_answer,
            "label_source": label_source,
            "confidence": f"{confidence:.3f}",
            "note": note,
        }
        for idx in range(1, 11):
            full_row[f"choice_{idx}"] = row[f"choice_{idx}"]
        full_rows.append(full_row)

        audit_row = {
            "id": qid,
            "final_answer": final_answer,
            "submission_fixed_answer": current_answer,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "num_runs": num_runs,
            "confidence": round(confidence, 3),
            "label_source": label_source,
            "note": note,
            "votes": votes,
        }
        audit_rows.append(audit_row)

        is_validated = label_source == "manual_kb_review" or (
            majority_answer is not None
            and final_answer == majority_answer
            and majority_count >= 5
        )

        if is_validated:
            validated_rows.append(full_row)
        else:
            review_rows.append(
                {
                    "id": qid,
                    "question": row["question"],
                    "current_answer": final_answer,
                    "submission_fixed_answer": current_answer,
                    "majority_answer": majority_answer,
                    "majority_count": majority_count,
                    "num_runs": num_runs,
                    "label_source": label_source,
                    "confidence": f"{confidence:.3f}",
                    "note": note,
                }
            )

    review_rows.sort(
        key=lambda item: (
            float(item["confidence"]),
            -int(item["majority_count"] or 0),
            int(item["id"]),
        )
    )

    cleaned_path = OUTPUT_DIR / "submission_fixed_clean.csv"
    full_path = OUTPUT_DIR / "train_labels_full_best_effort.csv"
    validated_path = OUTPUT_DIR / "train_labels_validated.csv"
    review_path = OUTPUT_DIR / "train_labels_review_queue.csv"
    audit_path = OUTPUT_DIR / "submission_fixed_audit.json"

    write_csv(cleaned_path, cleaned_rows, ["id", "answer"])
    write_csv(
        full_path,
        full_rows,
        ["id", "question"]
        + [f"choice_{idx}" for idx in range(1, 11)]
        + ["answer", "label_source", "confidence", "note"],
    )
    write_csv(
        validated_path,
        validated_rows,
        ["id", "question"]
        + [f"choice_{idx}" for idx in range(1, 11)]
        + ["answer", "label_source", "confidence", "note"],
    )
    write_csv(
        review_path,
        review_rows,
        [
            "id",
            "question",
            "current_answer",
            "submission_fixed_answer",
            "majority_answer",
            "majority_count",
            "num_runs",
            "label_source",
            "confidence",
            "note",
        ],
    )

    audit_summary = {
        "input_submission": str(SUBMISSION_FIXED_CSV.name),
        "cleaned_submission": str(cleaned_path.name),
        "manual_validations_count": len(MANUAL_VALIDATIONS),
        "validated_rows_count": len(validated_rows),
        "review_rows_count": len(review_rows),
        "manual_validation_ids": sorted(MANUAL_VALIDATIONS),
        "audit_rows": audit_rows,
    }
    audit_path.write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "cleaned_submission": str(cleaned_path.relative_to(ROOT)),
        "full_labels": str(full_path.relative_to(ROOT)),
        "validated_labels": str(validated_path.relative_to(ROOT)),
        "review_queue": str(review_path.relative_to(ROOT)),
        "audit_json": str(audit_path.relative_to(ROOT)),
        "validated_rows_count": len(validated_rows),
        "review_rows_count": len(review_rows),
    }


if __name__ == "__main__":
    result = build_outputs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
