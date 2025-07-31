# Akıllı Borç Tahsilat Stratejisi Ajanı (RL ile)

Bu proje, Pekiştirmeli Öğrenme (Reinforcement Learning - RL) kullanarak, finansal kurumlar için kârlılığı maksimize eden bir borç tahsilat stratejisi geliştirmektedir. Geliştirilen RL ajanı, her bir müşterinin durumuna (borç, gecikme günü vb.) göre en uygun iletişim kanalını (SMS, Arama, Bekleme) dinamik olarak seçmeyi öğrenir.

Proje, maliyet minimizasyonu ve tahsilat maksimizasyonu arasındaki dengeyi bulma problemine modern bir yapay zeka çözümü sunar.

## Projenin Çözdüğü Problem

Geleneksel borç tahsilat sistemleri genellikle statik kurallara dayanır. Bu proje ise her müşteriyi benzersiz bir vaka olarak ele alır ve şu sorulara dinamik yanıtlar bulur:
- Hangi müşteriyi aramalıyız, hangisine sadece bir hatırlatma yeterli?
- Ne zaman proaktif davranmalı, ne zaman beklemek daha kârlı?
- Müşteriyi kaybetme (churn) riskini göze alarak hangi adımı atmalıyız?

Ajan, bu kararları binlerce simülasyon üzerinden deneme-yanılma yoluyla öğrenerek en kârlı stratejiyi kendisi keşfeder.

## Teknolojiler ve Mimari

- **Programlama Dili:** Python
- **RL Kütüphaneleri:**
  - `Gymnasium`: Müşteri davranışlarını ve iş kurallarını simüle eden özel RL ortamı için.
  - `Stable-Baselines3`: Endüstri standardı Derin Q-Network (DQN) algoritmasını uygulamak için.
- **Veri Yönetimi:** `Pandas` & `Numpy`

Mimarinin kalbinde, müşteri personalarını ve bu personaların farklı iletişim kanallarına verdikleri tepkileri modelleyen bir simülasyon ortamı bulunmaktadır.

### RL Tasarımı

- **Durum (State):** `[Borç Miktarı, Gecikme Günü, Ödeme Geçmişi Skoru]`
- **Eylemler (Actions):** `[SMS Gönder, E-posta Gönder, Arama Yap, Bekle]`
- **Ödül Stratejisi (Reward):** Ajanın davranışını "akıllı kârlılık" yönünde şekillendiren, dikkatle tasarlanmış bir ödül-ceza mekanizması:
  - **Yüksek Ödül:** Başarılı tahsilat (özellikle arama ile yapılırsa ekstra bonus).
  - **Yüksek Ceza:** Müşteri kaybı (churn) veya borcun zaman aşımına uğraması.
  - **Düşük Maliyet:** Her iletişim denemesinin operasyonel maliyeti.

## Sonuçlar: Akıllı ve Kârlı Bir Ajan

Titiz bir "ödül mühendisliği" ve model eğitimi sonucunda, ajan son derece etkili ve kârlı bir strateji öğrenmeyi başarmıştır. 500 farklı müşteri senaryosunda yapılan testlerin sonuçları şöyledir:

| Metrik                  | Sonuç                       | Yorum                                                                                             |
| ----------------------- | --------------------------- | ------------------------------------------------------------------------------------------------- |
| **Ortalama Toplam Ödül**  | **+162.84**                 | Ajanın stratejisi yüksek derecede kârlıdır.                                                         |
| **Başarılı Tahsilat Oranı** | **%86.2**                   | Müşterilerin büyük çoğunluğundan borç başarılı bir şekilde tahsil edildi.                         |
| **Zaman Aşımı Oranı**       | **%4.4**                    | Pasif kalarak fırsatları kaçırma sorunu neredeyse tamamen ortadan kalktı.                          |
| **Müşteri Kaybı Oranı**   | **%9.4**                    | Ajan, genel kârlılık için hesaplanmış riskler almaktan çekinmiyor.                                |

### Ajanın Öğrendiği Strateji

- **Arama Yap (%69.1):** Ajan, "Arama" eyleminin en güçlü ve etkili araç olduğunu öğrendi ve bunu ana stratejisi olarak benimsedi.
- **Bekle (%27.0):** Herkese hemen saldırmak yerine, doğru anı beklemeyi de biliyor. Bu, stratejisinin "akıllı" olduğunu gösterir.
- **SMS Gönder (%3.9):** Sadece çok spesifik ve düşük riskli durumlarda kullanılan bir araç.

## Proje Yapısı ve Dosyalar

Proje, her biri belirli bir görevi yerine getiren modüler Python betiklerinden oluşur.

- **`DataFrame.py`**: Projenin temelini oluşturan `synthetic_customer_data.csv` dosyasını üretir. Gerçekçi senaryolar yaratmak için, farklı müşteri personalarına dayalı olarak mantıksal ve tutarlı veriler oluşturur.

- **`debt_collection_env.py`**: `Gymnasium` kütüphanesini kullanarak, RL ajanının etkileşimde bulunacağı özel simülasyon ortamını yaratır. Ajanın davranışını şekillendiren "oyun kuralları", ödül-ceza mekanizması ve müşteri davranışları bu dosyada tanımlanır.

- **`train.py`**: `Stable-Baselines3` kütüphanesini kullanarak Pekiştirmeli Öğrenme ajanını sıfırdan eğitir. Eğitim tamamlandığında, ajanın öğrendiği stratejiyi içeren `dqn_debt_collector.zip` adlı bir model dosyası oluşturur.

- **`evaluate.py`**: Eğitilmiş ajanın (`dqn_debt_collector.zip`) ne kadar etkili olduğunu test eder. Yüzlerce farklı müşteri senaryosunda performansını ölçer ve sonuçları konsola raporlar.

- **`dqn_debt_collector.zip`**: `train.py` tarafından oluşturulan, eğitilmiş ve kullanıma hazır model dosyasıdır.

- **`synthetic_customer_data.csv`**: `DataFrame.py` tarafından oluşturulan ve ajan tarafından eğitim ve test aşamalarında kullanılan sentetik müşteri veri setidir.

## Portfolyo Değeri

Bu proje, klasik sınıflandırma problemlerinin ötesine geçerek, dinamik ve stratejik bir iş problemini Pekiştirmeli Öğrenme ile modelleme yeteneğini sergilemektedir.

