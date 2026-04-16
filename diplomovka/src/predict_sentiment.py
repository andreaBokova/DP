import re
from pathlib import Path

import pandas as pd
import joblib
import nltk

from docx import Document

from preprocessing_helper import preprocess_sentence

# ___________________
# SETUP
# cesta k nespracovanym ESG reportom (.docx)
RAW_REPORTS = r"../data/esg/raw/KORPUS_LEI"

OUTPUT_CSV = r"results/esg_predictions/all_esg_predictions.csv"

# cesty k trom najlepsim modelom z Experimentu 2
MODEL_PATHS = {
    "model1_XGBoost_sentiment": r"results/experiment2/final_models/top1_XGBoost_pipeline.pkl",
    "model2_LinearSVC_sentiment": r"results/experiment2/final_models/top2_LinearSVC_pipeline.pkl",
    "model3_LogReg_sentiment": r"results/experiment2/final_models/top3_LogReg_pipeline.pkl",
}

# ______________
# NACITANIE DOCX

def load_docx_text(path: str) -> str:
    """
    nacita text z docx suboru
    """
    doc = Document(path)
    chunks = []

    # prejde vsetky odseky v dokumente
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            chunks.append(txt)

    # vsetky casti spoji do jedneho dlheho textu
    return "\n".join(chunks)


def split_to_sentences(text: str) -> list[str]:
    """
    rozdeli text na vety
    """
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences] # odstrani medzery okolo viet
    return sentences

# _________________
# EXTRAKCIA DAT DO MATICE

def extract_importance(folder_name: str) -> str:
    """
    ziska importance z nazvu suboru a formatuje (napr G_SIBS na G-SIBS)
    """
    name = folder_name.strip().upper()
    name = name.replace("_", "-")
    return name


def extract_year(filename_no_ext: str) -> str | None:
    """
    najde rok v nazve suboru (napr. annual_report_2024 vrati 2024)
    """
    match = re.search(r"(20\d{2})", filename_no_ext)
    return match.group(1) if match else None


def extract_bank_and_lei(folder_name: str) -> tuple[str | None, str | None]:
    """
    z nazvu foldera oddeli nazov banky a lei kod (priklad foldera> UBS_ 5299007QVIQ7IO64NX37)
    """
    folder_name = folder_name.strip()

    # najprv skusi regex, kde na konci je 20 znakovy kod
    match = re.match(r"^(?P<bank>.+?)_\s*(?P<lei>[A-Z0-9]{20})$", folder_name, flags=re.IGNORECASE)
    if match:
        bank = match.group("bank").strip()
        lei = match.group("lei").strip().upper()
        return bank, lei

    # ak to nevyjde, skusi rozdelit text podla posledneho podtrznika
    if "_" in folder_name:
        bank, lei = folder_name.rsplit("_", 1)
        return bank.strip(), lei.strip()

    # ak nie je lei, tak vratime nazov foldera
    return folder_name.strip(), None


def extract_metadata(file_path: Path, raw_reports: Path) -> dict:
    """
    z cesty k suboru vytiahne metadata
    ocakavana struktura je napr.
    RAW_REPORTS
        G_SIBS alebo L_SIBS alebo SIBS
            FRANCUZSKO
                BNP PARIBAS_ R0MUWSFPU8MPRO8K5P83
                    bnp_paribas_annual_report_2024_s681.docx
    """
    # zoberieme len cast cesty pod hlavnym priecinkom
    rel_parts = file_path.relative_to(raw_reports).parts

    # kontrola ci ma cesta ocakavanu strukturu
    if len(rel_parts) < 4:
        raise ValueError(
            f"Unexpected folder structure for file: {file_path}\n"
            f"Expected at least 4 relative parts got: {rel_parts}"
        )

    # jednotlive casti cesty
    importance_folder = rel_parts[0]
    country_folder = rel_parts[1]
    bank_lei_folder = rel_parts[2]
    filename = file_path.stem

    # ziskame banku, lei a rok
    bank, lei = extract_bank_and_lei(bank_lei_folder)
    year = extract_year(filename)

    # vratime metadata ako slovnik
    return {
        "year": year,
        "country": country_folder.strip(),
        "importance": extract_importance(importance_folder),
        "lei": lei,
        "bank": bank,
        "document": filename,
    }


# ____________________
# NACITANIE MODELOV
def load_models(model_paths: dict) -> dict:
    """
    nacita vsetky tri ulozene modely
    """
    models = {}
    for col_name, model_path in model_paths.items():
        print(f"Loading model: {model_path}")
        models[col_name] = joblib.load(model_path)
    return models


# ______________________________
# SPRACOVANIE JEDNEHO DOKUMENTU

def process_document(file_path: Path, raw_reports: Path, models: dict) -> list[dict]:
    """
    spracuje jeden dokument od nacitania az po predikciu
    vysledkom je zoznam riadkov do csv
    """
    # vytiahneme metadata z cesty suboru
    metadata = extract_metadata(file_path, raw_reports)

    # nacitame text z dokumentu
    raw_text = load_docx_text(str(file_path))

    # rozdelime text na vety
    sentences_raw = split_to_sentences(raw_text)

    # preprocessing viet rovnaky ako pri trenovani modelov
    processed_pairs = []
    for s in sentences_raw:
        cleaned = preprocess_sentence(s)

        # nechame len take vety ktore po preprocessingu nie su prazdne
        # a maju aspon 3 slova
        if cleaned and len(cleaned.split()) >= 3:
            processed_pairs.append((s, cleaned))

    # ak po preprocessingu nic nezostane vratime prazdny zoznam
    if not processed_pairs:
        return []

    # oddelime povodne vety a vycistene vety
    sentences_raw_filtered = [p[0] for p in processed_pairs]
    sentences_clean = [p[1] for p in processed_pairs]

    # na kazdu vetu pustime vsetky tri modely
    predictions = {}
    for col_name, model in models.items():
        preds = model.predict(sentences_clean)
        predictions[col_name] = preds

    # vytvorime vystupne riadky
    rows = []
    for i, sentence in enumerate(sentences_raw_filtered):
        row = {
            "year": metadata["year"],
            "country": metadata["country"],
            "importance": metadata["importance"],
            "lei": metadata["lei"],
            "bank": metadata["bank"],
            "document": metadata["document"],
            "sentence": sentence,
            "model1_XGBoost_sentiment": int(predictions["model1_XGBoost_sentiment"][i]),
            "model2_LinearSVC_sentiment": int(predictions["model2_LinearSVC_sentiment"][i]),
            "model3_LogReg_sentiment": int(predictions["model3_LogReg_sentiment"][i]),
        }
        rows.append(row)

    return rows


# ___________________________
# HLAVNY BEH PROGRAMU

def main():
    """
    hlavna funkcia ktora spusti cely proces
    """

    # nastavime hlavny vstupny priecinok a vystupny csv subor
    raw_reports = Path(RAW_REPORTS)
    out_csv = Path(OUTPUT_CSV)

    # ak vystupny priecinok neexistuje tak sa vytvori
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # nacitame modely
    models = load_models(MODEL_PATHS)

    # najdeme vsetky .docx subory vo vstupnom priecinku aj v podpriecinkoch
    docx_files = list(raw_reports.rglob("*.docx"))
    print(f"Found {len(docx_files)} .docx files")

    # sem budeme ukladat vysledky (a pripadne chyby)
    all_rows = []
    failed_files = []

    # postupne spracujeme kazdy dokument
    for idx, file_path in enumerate(docx_files, start=1):
        try:
            print(f"[{idx}/{len(docx_files)}] Processing: {file_path}")
            rows = process_document(file_path, raw_reports, models)
            all_rows.extend(rows)
        except Exception as e:
            print(f"FAILED: {file_path} -> {e}")
            failed_files.append({"file": str(file_path), "error": str(e)})

    # vytvorime dataframe
    df = pd.DataFrame(all_rows, columns=[
        "year",
        "country",
        "importance",
        "lei",
        "bank",
        "document",
        "sentence",
        "model1_XGBoost_sentiment",
        "model2_LinearSVC_sentiment",
        "model3_LogReg_sentiment",
    ])

    # ulozime vysledny csv subor
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved predictions to: {out_csv}")
    print(f"Total rows: {len(df)}")

    # ak nejake subory zlyhali ulozime ich do osobitneho csv
    if failed_files:
        failed_csv = out_csv.with_name("all_esg_predictions_failed_files.csv")
        pd.DataFrame(failed_files).to_csv(failed_csv, index=False, encoding="utf-8")
        print(f"Saved failed files log to: {failed_csv}")


if __name__ == "__main__":
    main()