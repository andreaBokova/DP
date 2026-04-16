from preprocessing_helper import (
    load_or_preprocess,
    add_sentence_key,
    LABEL_ORDER,
)

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier


# ______________________
# SETUP

DATASETS = {
    "100": "../data/financialphrasebank/raw/Sentences_AllAgree.txt",
}

RESULTS_DIR = "results/experiment2"
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

FINAL_MODEL_DIR = "final_model"
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)


# ____________________
# ZOSTAVENIE TRENOVACEJ A TESTOVACEJ MNOZINY

def build_train_and_test():
    """
    Train/Test split iba z datasetu AllAgree.
    Pouzivame 80 % na trenovanie a 20 % na testovanie.
    """

    # nacitanie datasetu
    df_100 = load_or_preprocess("100", DATASETS["100"])
    df_100 = add_sentence_key(df_100)

    # 80/20 split
    train_set, test_set = train_test_split(
        df_100,
        test_size=0.20,
        random_state=42,
        stratify=df_100["label"],
    )

    # reset index
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set


# _______________________
# MODELY
# modelom nastavujeme grid (mnozina hyperparametrov, ktore chceme vyskusat)

def get_models():
    # na zaklade vysledkov Experimentu 1 pouzivame pre vsetky BoW
    vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
    )

    models = {
        "MultinomialNB": (
            MultinomialNB(),
            {
                # clf__alpha je Laplace smoothing parameter, aby sa zabranilo nulovym pravdepodobnostiam
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
                # NB nepodporujeclass_weight
                # clf__fit_prior = False predpoklada,ze vsetky triedy su rovnako pravdepodobne
                # clf__fit_prior = True - model respektuje skutocne rozdelenie dat v triedach
                # v nasom pripade pri clf__fit_prior = True ma trieda neutral vyssiu pravdepodobnost este predtym ako sa zohladnia slova
                "clf__fit_prior": [True, False],
            },
        ),
        "ComplementNB": (
            ComplementNB(),
            {
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
                "clf__fit_prior": [True, False],
                # ComplementNB pocita vahy slov pre kazdu triedu, niektore slova mozu mat prilis velke vahy(riziko preucenia)
                # clf__norm:True normalizuje vahy (casto lepsia generalizacia)
                "clf__norm":[False, True],
                # 4×2×2=16 kombinaciix10 = 160 trenovani
            },
        ),
        "LinearSVC": (
            LinearSVC(max_iter=10000),
            {
                # GridSearchCV bude skusat vsetky kombinacie tychto hodnot
                # 0.1 znamena, ze silna regulacia, 5 znamena slabsia regulacia
                "clf__C": [0.1, 0.5, 1, 2, 5], 
                # clf__class_weight meni vahu chyby pocas ucenia
                # clf__class_weight=None znamena,ze vsetky triedy maju rovnaku vahu - chyba na netral ma rovnaky vyznam ako chyba na positive
                # clf__class_weight=None - model sa bude viac snazit minimalizovat chyby na vacsinovej triede
                # clf__class_weight=balanced znamena,ze vahy sa upravia podla velkosti tried
                # clf__class_weight=balanced - mensinova trieda ma vacsiu vahu - model viac tresta chyby na mensinovej triede
                "clf__class_weight": [None, "balanced"],
                # 5×2=10 kombinacii
                # 10×10=100 trenovani
            },
        ),
        "LogReg": (
            LogisticRegression(max_iter=2000),
            {
                "clf__C": [0.1, 0.5, 1, 2, 5],
                # algoritmus na optimalizaciu modelu,default je lbfgs-slaby, 
                # saga je casto lepsi pri BoW reprezentacii
                "clf__solver": ["lbfgs", "saga"], 
                "clf__class_weight": [None, "balanced"],
                # 5×2×2=20 kombinacii
                # 20×10=200 trenovani
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
            ),
            {
                "clf__n_estimators": [300, 600],
                # mac_depth None znamena,ze strom rastie, kym moze - riziko preucenia
                "clf__max_depth": [None, 30, 60],
                # kolko vzoriek v liste stromu - 1=strom je velmi specificky
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__class_weight": [None, "balanced"],
                # kolko [priznakov sa nahodne vyberie pri kazdom deleni stromu
                "clf__max_features": ["sqrt", "log2"],
                #2×3×3×2×2=72 komb.x10=720 trenovani
            },
        ),    
        "XGBoost": (
            XGBClassifier(
                #  model vracia pravdepodobnosti pre kazdu triedu
                objective="multi:softprob",
                # mame tri triedy
                num_class=3,
                # metoda, podla ktorej model hodnoti ucenie - multiclass log-loss
                eval_metric="mlogloss",
                # histogramova metoda trenovania stromov na CPU
                tree_method="hist",
                # pouziju sa vsetky dostupne jadra CPU 
                n_jobs=-1,
                # random state zabezpecuje reprodukovatelnost experimentu
                random_state=42,
            ),
            {
                # pocet stromov - viac stromov znamena silnejsi ale pomalsi model
                "clf__n_estimators": [300, 600, 900],
                # maximalna hlbka jedneho stromu (hlbsi model je komplexnejsi, ale riziko preucenia)
                "clf__max_depth": [3, 5, 7],
                # mensi leraning rate znamena pomalsie ucenie, ale casto lepsiu generalizaciu
                "clf__learning_rate": [0.05, 0.1],
                # kolko vzoriek sa nahodne pouzije pre kazdy strom
                # < 1.0 = menej preucenia
                "clf__subsample": [0.8, 1.0],
                # kolko priznakov (features) sa pouzije pre kazdy strom
                # < 1.0 = menej preucenia
                "clf__colsample_bytree": [0.8, 1.0],
                # minimalna vaha dat v liste, pravidlo na rast stromu
                # vyssia vaha - menej deleni
                "clf__min_child_weight": [1, 5],
                # L2 regularizacia vah v listoch
                # vyssia hodnota znamena silnejsiu regularizaciu - menej preucenia
                "clf__reg_lambda": [1.0, 3.0],

            },
        ),        
        "AdaBoost": (
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                random_state=42,
            ),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__learning_rate": [0.5, 1.0],
            },
            # 3×2=6 kombinaciix10=60 trenovani
        )
    }

    return models, vectorizer


# ________________________
# Spustenie EXPERIMENTU 2

def run_experiment2():
    # sledovanie najlepsich modelov
    best_model_overall = None
    best_model_name_overall = None
    best_test_macro_f1 = -1.0
    best_params_overall = None

    # priprava trenovacej a testovacej mnoziny
    train_df, test_df = build_train_and_test()

    print("\nDATASET STATISTICS")

    print("\nTRAIN SET")
    print("Total sentences:", len(train_df))
    print(train_df["label"].value_counts())

    print("\nTEST SET")
    print("Total sentences:", len(test_df))
    print(test_df["label"].value_counts())

    # vrati slovnik modelov a BoW vectorizer
    models, vectorizer = get_models()

    # texty po preprocessingu
    X_train = train_df["text_clean"]
    X_test = test_df["text_clean"]

    # triedy sentimentu
    y_train = train_df["label"]
    y_test = test_df["label"]

    # label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # tu sa bude ukladat summary tabulka
    all_summary_rows = []

    # sem si ulozime artefakty pre vsetky modely
    model_artifacts = {}

    # hlavna slucka pre kazdy model
    for name, (clf, grid) in models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")

        pipe = Pipeline([
            ("vec", vectorizer),
            ("clf", clf),
        ])

        # ladenie hyperparametrov
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=10,          # 10-nasobna krizova validacia
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train_enc)

        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_best = float(search.best_score_)

        # finalne testovanie
        y_pred_enc = best_model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        # zapamataj si vitazny model podla TEST macro F1
        if test_macro_f1 > best_test_macro_f1:
            best_test_macro_f1 = test_macro_f1
            best_model_overall = best_model
            best_model_name_overall = name
            best_params_overall = best_params

        print(f"Best CV macro-F1: {cv_best:.4f}")
        print(f"Best params: {best_params}")
        print(f"TEST macro-F1 (AllAgree20): {test_macro_f1:.4f}")

        # report + confusion matrix
        report = classification_report(
            y_test,
            y_pred,
            digits=4,
            labels=LABEL_ORDER
        )
        cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

        # ulozime artefakty do pamate, aby sme ich vedeli neskor filtrovat na top 3
        model_artifacts[name] = {
            "cv_best": cv_best,
            "test_macro_f1": test_macro_f1,
            "best_params": best_params,
            "report": report,
            "confusion_matrix": cm,
            "best_model": best_model,
        }

        all_summary_rows.append({
            "model": name,
            "cv_best_macro_f1": cv_best,
            "test_macro_f1": test_macro_f1,
            "best_params": str(best_params),
        })

    # summary vsetkych modelov
    summary_df = pd.DataFrame(all_summary_rows).sort_values("test_macro_f1", ascending=False)
    summary_path = os.path.join(RESULTS_DIR, "experiment2_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # top 3 modely
    top3_df = summary_df.head(3).copy()
    top3_summary_path = os.path.join(RESULTS_DIR, "experiment2_top3_summary.csv")
    top3_df.to_csv(top3_summary_path, index=False)

    print(f"\nSaved full summary to: {summary_path}")
    print(f"Saved top 3 summary to: {top3_summary_path}")

    # ulozenie classification reportov len pre top 3 modely
    for model_name in top3_df["model"]:
        artifact = model_artifacts[model_name]
        report_path = os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {model_name}\n")
            f.write(f"BEST CV macro-F1: {artifact['cv_best']:.6f}\n")
            f.write(f"TEST macro-F1: {artifact['test_macro_f1']:.6f}\n")
            f.write(f"BEST PARAMS: {artifact['best_params']}\n\n")
            f.write("=== Classification Report (TEST) ===\n")
            f.write(artifact["report"])
            f.write("\n\n=== Confusion Matrix (rows=true, cols=pred) ===\n")
            f.write(str(artifact["confusion_matrix"]))

    # ulozenie len top 3 modelov
    for rank, model_name in enumerate(top3_df["model"], start=1):
        artifact = model_artifacts[model_name]
        model_path = os.path.join(FINAL_MODEL_DIR, f"top{rank}_{model_name}_pipeline.pkl")
        info_path = os.path.join(FINAL_MODEL_DIR, f"top{rank}_{model_name}_info.txt")

        joblib.dump(artifact["best_model"], model_path)

        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"RANK: {rank}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"BEST CV macro-F1: {artifact['cv_best']:.6f}\n")
            f.write(f"TEST macro-F1: {artifact['test_macro_f1']:.6f}\n")
            f.write(f"BEST PARAMS: {artifact['best_params']}\n")

    # label encoder staci ulozit raz
    joblib.dump(le, os.path.join(FINAL_MODEL_DIR, "label_encoder.pkl"))

    # info o vitazovi
    with open(os.path.join(FINAL_MODEL_DIR, "best_model_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"BEST MODEL: {best_model_name_overall}\n")
        f.write(f"TEST macro F1: {best_test_macro_f1:.6f}\n")
        f.write(f"BEST PARAMS: {best_params_overall}\n")

    print("\nTOP 3 MODELS SAVED")
    print(f"Vitazny model: {best_model_name_overall} | TEST macro F1: {best_test_macro_f1:.4f}")
    print(f"Top 3 modely ulozene v: {FINAL_MODEL_DIR}")
    print(f"Top 3 classification reports ulozene v: {REPORTS_DIR}")
    print(f"Label encoder ulozeny v: {os.path.join(FINAL_MODEL_DIR, 'label_encoder.pkl')}")


if __name__ == "__main__":
    run_experiment2()