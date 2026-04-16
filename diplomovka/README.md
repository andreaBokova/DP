# SPUSTENIE SKRIPTOV

## 1. Poziadavky
* Python 3.10 - 3.12 (odporucane, pri nizsej verzii moze dojst k runtime chybam)
* Visual Studio Code alebo ine IDE podporujuce Python

## 2. Vytvorenie a aktivacia virtualneho prostredia (venv)
* vo Windows (PowerShell) spustite nasledujuce prikazy:
  python -m venv .venv<br>
  ..venv\Scripts\Activate.ps1

## 3. Instalacia zavislosti:
   pip install --upgrade pip<br>
   pip install -r requirements.txt

## 4. Presun do priecinka src, kde sa nachadzaju Python skripty: 
   cd src 

## 5. Spustenie Experimentu 1
   python experiment_1.py

## 6. Spustenie Experimentu 2
   python experiment_2.py

## 7. Spustenie predict_sentiment.py
   python predict_sentiment.py

# INFORMACIE O SKRIPTOCH

## experiment_1.py:

* skript nacita a predspracuje datasety 50Agree, 66Agree, 75Agree a AllAgree,
* z AllAgree vytvori fixny test set (20 %),
* odstrani prekryv viet medzi train a test mnozinou,
* pre kazdy train variant spusti kombinacie:
   * reprezentacia: bow alebo tfidf
   * trening: balanced alebo unbalanced
* natrenuje Logistic Regression model,
* vyhodnoti model cez F1, precision, recall a confusion matrix
* vysledky ulozi do priecinku results/experiment1

## experiment_2.py

* skript nacita a predspracuje dataset AllAgree,
* rozdeli ho na treningovu a testovaciu mnozinu v pomere 80 % / 20 %,
* pouzije reprezentaciu bow,
* pre kazdy klasifikacny model spusti GridSearchCV a 10-nasobnu krizovu validaciu, natrenuje najlepsiu verziu kazdeho modelu, vyhodnoti model cez macro F1, classification report a confusion matrix,
* porovna vsetky modely a vyberie top 3 najlepsie,
* vysledky ulozi do priecinkov results/experiment2 a final_model

## predict_sentiment.py

* skript nacita .docx ESG reporty z priecinka data/esg/raw/KORPUS_LEI,
* z cesty a nazvu suboru extrahuje metadata o dokumente (rok, krajina, dolezitost, LEI, banka, nazov dokumentu),
* text kazdeho dokumentu rozdeli na vety,
* na kazdu vetu aplikuje rovnaky preprocessing ako pri trenovani modelov,
* ponecha len neprazdne vety s aspon 3 slovami,
* nacita 3 najlepsie modely z experimentu 2,
* pre kazdu vetu vytvori sentimentovu predikciu pomocou vsetkych 3 modelov,
* vysledky ulozi do suboru results/esg_predictions/all_esg_predictions.csv

## preprocessing_helper.py

* skript obsahuje pomocne funkcie na nacitanie a predspracovanie datasetu Financial PhraseBank,
* nacita .txt subory, kde kazdy riadok obsahuje vetu a sentimentovy label,
* vykona zakladne cistenie textu,
* aplikuje tokenizaciu, odstranenie stop slov a lematizaciu pomocou Stanza,
* odstrani prazdne, prilis kratke a duplicitne vety,
* vytvori stlpec text_clean, ktory sa pouziva ako vstup pre modely,
* vyuziva cache mechanizmus, aby sa uz raz spracovane datasety nemuseli preprocessovat znova,
* vytvara pomocny kluc vety na porovnavanie train a test mnoziny,
* a obsahuje funkciu na odstranenie prekryvu viet medzi train a test datami.

## dataset_stats.py:

* skript nacita surovy dataset Financial PhraseBank (.txt),
* oddeli vetu a label
* vykona tokenizaciu viet,
* vypocita zakladne statistiky datasetu:
  * pocet viet,
  * velkost slovnej zasoby (unikatne tokeny),
  * distribuciu tried sentimentu,
  * priemerny pocet tokenov vo vete,
  * priemerny pocet znakov vo vete,
*  statistiky vypise do konzoly.
