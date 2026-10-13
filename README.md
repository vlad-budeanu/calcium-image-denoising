# Reducerea zgomotului în imagistica cu calciu (Calcium Imaging Denoising)

## 1. Descriere Generală
Proiectul își propune eliminarea zgomotului (Poisson și Gaussian) din secvențe video de imagistică cu calciu folosind o abordare bazată pe rețele neuronale convoluționale (arhitectura U-Net) și compararea acesteia cu o metodă clasică de filtrare (OpenCV Fast Non-Local Means).

---

## 2. Analiza Literaturii de Specialitate (State-of-the-Art)
# Calcium Imaging Denoising Methods (2020–2025)


| Autor(i) / An | Titlul articolului / proiectului | Tehnologie / Algoritm | Metodologie / Abordare | Rezultate / Performanță | Limitări | Comentarii suplimentare |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Li et al. (2021)** *(DeepCAD)* | *Reinforcing neural spike detection in deep-tissue calcium imaging with 3D self-supervised denoising* | Rețele Convoluționale 3D (CNN), Deep Learning Self-Supervised | Antrenarea unei rețele fără imagini "ground truth" curate, folosind cadre temporale 3D adiacente pentru eliminarea zgomotului Poisson. | Creșterea SNR-ului de >10 ori, reducerea laserului cu până la 90%, detectarea spike-urilor cu o acuratețe de >95%. | Poate introduce mici artefacte vizuale la mișcări elastice extreme ; necesită GPU-uri puternice. | Open-source. Permite imagistică la puteri laser foarte mici. |
| **Lecoq et al. (2021)** *(DeepInterpolation)* | *Removing independent noise in biological microscopy with DeepInterpolation* | Deep Learning (U-Net adaptat spațio-temporal) | Prezice cadrul pe baza cadrelor adiacente din trecut și viitor, eliminând zgomotul independent temporal. | Creșterea SNR-ului de >6 ori pe date cu zgomot Poisson și de citire CMOS. | Poate subestima zgomotul corelat temporal pe perioade lungi; procesare mai lentă per cadru. | Funcționează excelent pe serii temporale 2D de mari dimensiuni. |
| **Giovannucci et al. (2020/2022)** *(CaImAn / CNMF-E)* | *CaImAn an open source pipeline for calcium imaging data analysis* | Descompunere Matriceală Non-Negativă (CNMF-E) + CNN | Descompune matricea de date brute în componente celulare (spațiale) și semnale de calciu (temporale). | Standardul industrial pentru izolarea semnalului de fundal; acuratețe de >90% în extragerea urmelor de calciu fără neuropil. | Sensibil la variații mari de iluminare de fond; necesită ajustarea manuală a parametrilor inițiali. | Soluție clasică/hibridă. Nu este un "black box" ca AI-ul pur, oferind transparență matematică totală. |
| **Buchanan et al. (2022)** *(DeepVID)* | *Deep Learning for Real-Time Video Denoising in Optical Imaging* | Autoencodere Convoluționale Temporale | Antrenament self-supervised pe secvențe video rapide pentru reducerea zgomotului la nivel de pixel în timp real. | Denoising la frecvențe de >100 Hz cu latență de procesare sub 10 ms per cadru. | Compromis la calitatea spațială comparativ cu modelele 3D mai mari care procesează offline. | Optim pentru experimente în timp real unde latența scăzută este critică. |
| **Guan et al. (2024)** *(ST-Transformer)* | *Spatio-Temporal Transformer for Ultra-Low Light Calcium Imaging Denoising* | Vision Transformers (ViT) Spațio-Temporali | Utilizează mecanisme de atenție (Self-Attention) pentru a corela informația din pixeli îndepărtați spațial și temporal. | Reconstrucție precisă a semnalului de calciu la nivel de foton unic (ultra-low light imaging). | Risc de over-smoothing dacă atenția nu este calibrată; cerințe extreme de memorie VRAM. | Surprinde dependențe pe distanțe lungi mult mai bine decât rețelele convoluționale tradiționale. |

---


## 3. Arhitectura Soluției

### Schema Bloc
![Arhitectura Soluției](docs/schema_bloc.png)

### Descrierea Arhitecturii Soluției

- **Date de Intrare (Data Ingestion):** Recepționează secvențele video de imagistică cu calciu în format `.tif` sau `HDF5` și le încarcă în memorie sub formă de matrici 3D cu ajutorul bibliotecii `tifffile`.

- **Modulul de Preprocesare Date:** Realizează normalizarea valorilor pixelilor în intervalul `[0, 1]`. Generează zgomot sintetic (Gaussian/Poisson via `NumPy`) adăugat peste imaginile originale pentru a crea perechile de antrenare: imagini cu zgomot (`X`) și ținta curată / Ground Truth (`y`). Împarte setul de date în 80% antrenare și 20% testare.

- **Modulul de Filtrare Clasică (OpenCV Baseline):** Aplică algoritmul clasic `cv2.fastNlMeansDenoising()` pe setul de test pentru a stabili o linie de referință (baseline).

- **Modulul Deep Learning (Rețeaua U-Net):** Implementează o rețea neuronală convoluțională bazată pe arhitectura U-Net în `TensorFlow/Keras`. Conține o etapă de reducere a dimensiunii (*Encoder*), o etapă de reconstrucție (*Decoder*) și conexiuni directe (*Skip Connections*) pentru a conserva detaliile morfologice fine ale neuronilor.

- **Modulul de Evaluare & Metrici:** Calculează metrici matematice obiective de calitate a imaginii — **PSNR** și **SSIM** — folosind pachetul `scikit-image`, comparând rezultatele oferite de metoda clasică și metoda Deep Learning raportat la Ground Truth.

- **Modulul de Vizualizare:** Afișează o grilă comparativă a imaginilor procesate și randează graficele cu curbele de învățare (*Loss Train vs. Validation*) folosind `Matplotlib`.
