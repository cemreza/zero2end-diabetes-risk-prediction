Diyabet Risk Tahmin Projesi

Problemin Tanımı
Biyoloji lisans eğitimim ve biyoteknoloji alanındaki iş geçmişimi harmanlamaya çalıştığım bir bootcamp projesi yapmaya çalıştım. Bunun için ise bootcamp proje şartlarını kapsadığını düşündüğüm bi diyabet data seti ile çalıştım.
Diyabet dünya genelinde oldukça artış gösteren ve erken tanımı önemli olan bir hastalık olarak görülmektedir. Diyabetin tanımını kolaylaştıracak ve hızlandıracak bir makine öğrenmesi geliştirmek istedim ve bunun için kaggledaki kapsamlı bir data setin kullanarak gerçekleştirdim ve kişilerin diyabet geliştirme riskini tahmin eden basit ama işlevsel bir makine öğrenmesi uygulaması geliştirmeye çalıştım. Amacım; veri analizinden model eğitimine, hiperparametre optimizasyonundan Streamlit arayüzüne kadar uçtan uca bir ML projesi deneyimi yaratmak ve bunları gerçek bir ürün akışına benzeterek öğrenmekti.

Projenin Amacı
Makine öğrenmesi tekniklerini kullanarak:
Diyabet riskini etkileyen temel faktörleri anlamak
Bu faktörleri modele uygun hale getirmek
Bir sınıflandırma modeli eğitmek
Tahminleri kullanıcı dostu bir arayüzle sunmak


Pipeline:

-Veri Seti Seçimi
Projede kaggledan bulduğum, CDC’nin geniş halk sağlığı tarama verilerinden (BRFSS) üretilmiş bir diyabet sınıflandırma veri setini kullanıyor.
Veri seti büyük olduğu için model eğitimini hızlandırmak adına temsil gücünü kaybetmeyecek küçük bir örneklem oluşturdum.
Veri setinde BMI, kan basıncı, kolesterol takibi, fiziksel aktivite alışkanlıkları, sağlık durumu gibi günlük yaşama dair birçok özellik bulunduran geniş kapsamlı bir veri setiydi.
Büyük veri boyutu nedeniyle %10'luk temsil gücü yüksek bir örneklem kullanıldı.

-EDA (Keşifsel Veri Analizi)
Değişken dağılımları
Korelasyon matrisi
Sınıf dengesizliği
Sağlık göstergeleri arasındaki ilişkiler

-Feature Engineering
Sürekli değişkenlerin ölçeklenmesi
Kategorik değişkenlerin encode edilmesi
Model için son özellik setinin oluşturulması

-Baseline Model
Logistic Regression ile başlangıç performansı alındı.

-Hyperparameter Optimization
RandomizedSearchCV kullanılarak en iyi parametre seti arandı.

-Model Değerlendirme
Accuracy
Precision, Recall, F1
ROC-AUC

-Modelin Kaydedilmesi
final_model.pkl olarak dışa aktarıldı.

-Streamlit Uygulaması
Model sadece notebook ortamında kalmasın diye Streamlit kullanarak etkileşimli bir web arayüzü oluşturdum.
Uygulama, kullanıcının girdiği sağlık bilgilerini hemen işleyip tahmini risk yüzdesini üretir.

/proje yapısı/

zero2end-diabetes-risk-prediction/

│── README.md

│── requirements.txt

│── data/

│   ├── raw/ (orijinal veri seti)

│   └── processed/ (örneklem ve işlenmiş veriler)

│── notebook/

│   ├── 01_EDA.ipynb

│   ├── 02_baseline.ipynb

│   ├── 03_feature_engineering.ipynb

│   ├── 04_model_optimization.ipynb

│   ├── 05_evaluation.ipynb

│── models/

│   └── final_model.pkl

│── src/

│   ├── preprocess.py

│   ├── train.py

│   ├── inference.py

│── app/

│   └── streamlit_app.py


Sonuç olarak;
Model ROC-AUC değerinin 0.81 olması, diyabet riski taşıyan ve taşımayan bireyler arasında ayırt etme gücünün oldukça iyi olduğunu gösteriyor.
Sağlık tarama modellerinde AUC ≥ 0.80 seviyeleri uygulamada kullanılabilir kabul edilmektedir.
Modelin yüksek recall değeri diyabet riskli bireyleri kaçırmadığını gösterdim.

Model genel olarak:
Basit, yorumlanabilir
Hafif ve hızlı
Gerçek zamanlı inference verebilen bir yapı sunmaya çalıştım.


