----Product Category Classifier----

Proiect realizat in cadrul cursului **Machine Learning - Link Academy**.  
Scopul acestui proiect este dezvoltarea unui model de invatare automata care sa **prezica automat categoria unui produs** pe baza titlului sau (`Product Title`).

Modelul a fost antrenat folosind setul de date `products.csv`, care contine peste 30.000 de produse si categoriile lor.

---

---Scopul proiectului---

In comertul online, mii de produse noi sunt adaugate zilnic, iar clasificarea manuala este lenta si predispusa la erori.  
Acest proiect propune o solutie automata: un model ML care sugereaza imediat categoria potrivita pe baza titlului produsului.

Prin acest sistem:
- se reduce timpul de introducere a produselor in platforma,
- se minimizeaza erorile umane,
- se imbunatateste experienta utilizatorilor in cautare si filtrare.

---

---Structura proiectului---

Proiect_Final/

 -models
	->category_model.pkl -> Modelul final antrenat (TF-IDF + LinearSVC)

 -products.csv -> Setul de date folosit la antrenare
 -train_model.py -> Script pentru antrenarea si salvarea modelului
 -predict_category.py -> Script pentru testarea interactiva a modelului
 -Product_Category_Classifier.ipynb -> Notebook complet documentat (analiza si comparatii)
 -README.md # Acest fisier



---

---Cum rulezi proiectul---

1. Antrenarea modelului

Ruleaza in terminal din folderul proiectului:

"python train_model.py --data products.csv --model_out models/category_model.pkl"
Modelul foloseste un pipeline TF-IDF + LinearSVC si este salvat automat in folderul "models", care ar trebui inclus in folderul proiectului.


2. Testarea modelului
Dupa ce modelul este antrenat:

"python predict_category.py --model models/category_model.pkl"
Programul iti va cere un titlu de produs:

Titlu produs: bosch serie 4 kgv39vl31g
Rezultatul va fi afisat sub forma:

Categoria prezisa: Fridge Freezers

Top 5 categorii probabile:
 - Fridge Freezers (score: -0.48)
 - Fridges (score: -0.61)
 - Dishwashers (score: -0.72)
 - Washing Machines (score: -0.81)
 - Microwaves (score: -0.90)

---Tehnologii folosite---
Python 3.12

Pandas

Scikit-learn

Joblib

Jupyter Notebook

---Rezultate si evaluare---
In cadrul notebook-ului Product_Category_Classifier.ipynb au fost testate mai multe modele de clasificare:

Model	Acuratete
Logistic Regression	0.9488
Linear SVC	0.9498

Modelul LinearSVC a obtinut cea mai buna performanta si l-am ales ca model final.
Matricea de confuzie arata o clasificare corecta pentru majoritatea categoriilor, cu mici confuzii intre clase similare (ex: Fridges si Freezers).

---Concluzii---
Modelul LinearSVC este cea mai buna alegere pentru clasificarea titlurilor de produse.

Performanta generala depaseste 94-95% acuratete.

Poate fi integrat usor intr-un sistem real de e-commerce pentru etichetarea automata a produselor.

Extensii viitoare: calibrarea probabilitatilor, imbunatatirea vectorizarii TF-IDF si adaugarea altor feature-uri (brand, numar de caractere, majuscule etc.)

Autor
[Partenie Alexandru]
Cursant - Link Academy
Noiembrie 2025
