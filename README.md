# evostream: Ewolucyjna Samoadaptacja Potoków NLP 

> **Praca Magisterska:** *Ewolucyjna samoadaptacja potoków NLP: Wielokryterialna optymalizacja klasteryzacji strumieniowej tekstu* 
> **Autor:** Jakub Walkowicz 
> **Promotor:** dr Beata Basiura 
> **Uczelnia:** Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie (AGH) 

---

## Spis Treści 
1. [Przegląd Projektu ](#przegląd-projektu)
2. [Główne Tezy Badawcze ](#główne-tezy-badawcze)
3. [Architektura Systemu ](#architektura-systemu)
4. [Struktura Repozytorium ](#struktura-repozytorium)
5. [Instalacja i Wymagania ](#instalacja-i-wymagania)
6. [Szybki Start: Interaktywny Panel Demonstracyjny ](#szybki-start-interaktywny-panel-demonstracyjny)
7. [Uruchomienie Potoku Strumieniowego ](#uruchomienie-potoku-strumieniowego)
8. [Reprodukcja Wyników i Wykresów do Pracy ](#reprodukcja-wyników-i-wykresów-do-pracy)

---

## Przegląd Projektu 

System **evostream** realizuje nienadzorowaną, dwufazową klasteryzację ewoluujących strumieni dokumentów tekstowych w czasie rzeczywistym. Wykorzystuje:
- **Głębokie osadzenia semantyczne (SBERT)** sprzężone z **Inkrementalną Analizą Głównych Składowych (IPCA)** do redukcji wymiarowości w locie ($O(1)$ latency).
- **Dwufazowe grupowanie strumieniowe:** Faza *online* oparta na mikroklastrach (`DenStream`) oraz faza *offline* agregująca centra mikroklastrów w makrotematy (`DBSCAN`).
- **Nienadzorowany detektor dryfu:** Ciągłe monitorowanie degradacji spójności klastrów ($Z$-score na wskaźniku *Silhouette*) oraz wskaźnika proliferacji bufora obserwacji odstających ($R_{\text{outlier}}$).
- **Wielokryterialny algorytm ewolucyjny (NSGA-II):** Samoczynne wyznaczanie frontu Pareto (kompromis: *Jakość semantyczna* vs. *Złożoność obliczeniowa*) i automatyczna adaptacja hiperparametrów ($\epsilon, \mu, \lambda, \epsilon_{\text{offline}}$) w wyznaczonym geometrycznie **punkcie kolana (punkt przegięcia)**.

---

## Główne Tezy Badawcze 

| Teza | Treść Tezy | Status Weryfikacji | Kluczowy Wynik Empiryczny |
| :--- | :--- | :---: | :--- |
| **Teza 1** | Integracja osadzeń SBERT z inkrementalną redukcją IPCA zachowuje czystość klastrów przy drastycznym skróceniu czasu przetwarzania ($O(1)$). | **POTWIERDZONA** | **$8.17\times$ przyspieszenie** online ($d=16$), wzrost czystości z $19.3\%$ do **$57.7\%$** (przełamanie klątwy wymiarowości). |
| **Teza 2** | Wielokryterialna optymalizacja NSGA-II umożliwia skuteczną, tzw. „ślepą” adaptację parametrów do dryfu bez ingerencji człowieka. | **POTWIERDZONA** | Samoczynne wykrycie dryfu i rekonfiguracja $\epsilon, \lambda$; odzyskanie czystości ($51.1\% - 58.5\%$) bez przestojów. |
| **Teza 3** | Istnieje stabilny kompromis Pareto między jakością a złożonością, stabilizujący się w punkcie kolana przy niskich wymiarach projekcji. | **POTWIERDZONA** | Zbieżność punktu przegięcia w zakresie niskich wymiarów ($d=16-32$) przy minimalnym koszcie obliczeniowym. |

---

## Architektura Systemu 

```
 [ Apache Kafka / 20 Newsgroups Stream ]
 │
 ▼
 [ Text Preprocessing & Cleaning ]
 │
 ▼
 [ SBERT Embeddings (all-MiniLM-L6-v2) ]
 │
 ▼
 [ Incremental PCA (IPCA, d = 16) ]
 │
 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ FAZA ONLINE: DenStream Micro-Clustering │
│ │
│ • P-Micro-Clusters: CF1, CF2, N, w(t) = w0 * 2^(-λΔt) │
│ • O-Micro-Clusters: Bufor obserwacji odstających (Outlier Buffer) │
└────────────────────────────────────────────────────────────────────────────────────────┘
 │ │
 ▼ (Centroidy mikroklastrów) ▼ (Metryki: Sil, Outlier Ratio)
┌──────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ FAZA OFFLINE: DBSCAN Macro │ │ UNSUPERVISED DRIFT DETECTOR │
│ │ │ │
│ Agregacja mikroklastrów w makro- │ │ 1. Degradacja Silhouette: Sil < μ - 2σ │
│ klastry tematyczne o promieniu │ │ 2. Proliferacja odstających: Rout > 30% │
│ ε_offline │ └───────────────────────────────────────────┘
└──────────────────────────────────────┘ │
 │ (Gdy dryf wykryty)
 │ │
 ▼ ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ TELEMETRIA I BAZA POSTGRESQL │ │ OPTYMALIZATOR EWOLUCYJNY (NSGA-II) │
│ │ │ │
│ Purity, Silhouette, Davies-Bouldin, │◄────┤ • Cel 1: Jakość (Max Silhouette) │
│ ARI, NMI, Latency, RAM, Ratio │ │ • Cel 2: Złożoność (Min Micro-Clusters) │
│ │ │ ──► Wyznaczenie Frontu Pareto & Punkt przegięcia │
└──────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## Struktura Repozytorium 

```
evostream-ga-sentiment/
├── src/
│ ├── apps/
│ │ ├── daemon.py # Główny demon przetwarzania strumieniowego
│ │ ├── ingesting.py # Generator strumienia Kafka ze wzorcami dryfu
│ │ ├── setup.py # Inicjalizacja tematów Kafki i tabel PostgreSQL
│ │ └── web_ui.py # Interaktywny panel Streamlit do obrony pracy
│ ├── core/
│ │ ├── config.py # Konfiguracja Pydantic dla środowiska
│ │ └── logger.py # Loguru logger
│ ├── domain/
│ │ ├── clustering.py # Dwufazowy klasteryzator (DenStream + DBSCAN + metryki)
│ │ ├── drift.py # Nienadzorowany detektor dryfu koncepcji
│ │ ├── evolution.py # Wielokryterialny optymalizator NSGA-II i punkt przegięcia
│ │ └── preprocessing.py # Czyszczenie tekstu i transformacja SBERT + IPCA
│ ├── infrastructure/
│ │ ├── kafka/client.py # Klienci producenta i konsumenta Kafka
│ │ └── postgres/client.py # Klient bazy danych PostgreSQL
│ ├── model/
│ │ └── schemas.py # Schematy bazy danych i modeli telemetrycznych
│ └── main.py # Główny interfejs wiersza poleceń CLI (Typer)
│
├── experiments/ # Eksperymenty i tabele LaTeX do Rozdziału 4
│ ├── exp_thesis_1_ipca.py # Eksperyment dla Tezy 1 (IPCA vs Pełna przestrzeń SBERT)
│ ├── exp_thesis_2_drift.py # Eksperyment dla Tezy 2 (Reaktywna adaptacja dryfu)
│ ├── exp_thesis_3_pareto.py # Eksperyment dla Tezy 3 (Fronty Pareto i punkt przegięcia)
│ ├── thesis_1_table.tex # Gotowa tabela LaTeX dla Tezy 1
│ ├── thesis_2_table.tex # Gotowa tabela LaTeX dla Tezy 2
│ └── thesis_3_table.tex # Gotowa tabela LaTeX dla Tezy 3
│
├── charts/ # Wygenerowane wykresy publikacyjne (300 DPI PNG & PDF)
│ ├── thesis_1_ipca_benchmark.png / .pdf
│ ├── thesis_2_drift_adaptation.png / .pdf
│ ├── thesis_3_pareto_tradeoff.png / .pdf
│ └── regenerate_all_thesis_charts.py
│
├── config/
│ ├── base.yaml # Konfiguracja bazowa
│ ├── test.yaml # Konfiguracja środowiska testowego
│ └── prod.yaml # Konfiguracja środowiska produkcyjnego
│
├── docker-compose.yml # Kontenery Kafka, Zookeeper, PostgreSQL
├── pyproject.toml # Zależności i konfiguracja projektu UV
└── README.md # Dokumentacja techniczna i naukowa
```

---

## Instalacja i Wymagania 

### Wymagania:
- **Python 3.11+**
- Menedżer pakietów **`uv`** ([Instrukcja uv](https://github.com/astral-sh/uv))
- **Docker** i **Docker Compose** (dla Kafki i PostgreSQL)

### 1. Klonowanie i instalacja środowiska:
```bash
cd evostream-ga-sentiment
uv sync
```

### 2. Uruchomienie infrastruktury (Kafka + PostgreSQL):
```bash
docker compose up -d
```

### 3. Inicjalizacja tabel i tematów:
```bash
uv run python -m src.main setup
```

---

## Szybki Start: Interaktywny Panel Demonstracyjny 

Do prezentacji działania systemu podczas **obrony pracy magisterskiej przed komisją** przygotowano dedykowany, interaktywny panel webowy:

```bash
uv run streamlit run src/apps/web_ui.py
```
*Aplikacja otworzy się automatycznie w przeglądarce pod adresem: `http://localhost:8501`*

### Co można zaprezentować komisji na żywo w panelu?
1. ** Następna partia (Kolejna partia dokumentów):** Realny napływ dokumentów ze strumienia, ich osadzanie SBERT i redukcja IPCA w czasie rzeczywistym.
2. ** Przestrzeń Semantyczna (2D Map):** Interaktywna mapa rozkładu dokumentów z zaznaczonymi w czasie rzeczywistym **centrami mikroklastrów** i ich promieniami.
3. ** Wywołaj dryf! (Wymuś dryf pojęć):** Przycisk natychmiastowego przełączenia domeny tematycznej (np. z *Nauka/Kosmos* na *Polityka/Bliski Wschód*).
4. ** Reaktywna detekcja:** Prezentacja, jak wskaźnik obserwacji odstających ($R_{\text{outlier}} > 30\%$) i spadek Silhouette natychmiast wykrywają zmianę rozkładu.
5. ** Samoadaptacja NSGA-II:** Wizualizacja wygenerowanego **frontu Pareto** oraz automatycznego doboru optymalnego **punktu przegięcia**.
6. ** Tryby adaptacji:** Możliwość przełączania w locie pomiędzy adaptacją *modyfikacja parametrów w miejscu* a przeładowaniem modelu *wymiana w locie*.
7. ** Wyniki do pracy:** Przegląd wykresów i tabel z eksperymentów do Rozdziału 4.

---

## Uruchomienie Potoku Strumieniowego 

Pełny potok produkcyjny oparty na kolejce Kafka i demonie przetwarzania uruchamia się za pomocą dwóch poleceń CLI:

### Terminal 1: Uruchomienie Demona Klasteryzacji
```bash
# Uruchamia demona przetwarzającego strumień z nienadzorowaną detekcją dryfu
uv run python -m src.main daemon --use-pca --pca-dim 16
```

### Terminal 2: Uruchomienie Ingestora Danych z Wstrzykiwaniem Dryfu
```bash
# Wysyła wiadomości do Kafki z symulacją nagłego dryfu koncepcji po 1500 dokumentach
uv run python -m src.main ingest --interval 0.2 --drift-type sudden --drift-step 1500
```

---

## Reprodukcja Wyników i Wykresów do Pracy 

Wszystkie eksperymenty opisane w Rozdziale 4 pracy są w pełni deterministyczne i powtarzalne:

### 1. Teza 1: Benchmark IPCA vs Pełna Przestrzeń SBERT (384d)
```bash
uv run python -m src.main benchmark-thesis-1
```
*Generuje: `charts/thesis_1_ipca_benchmark.png/.pdf` oraz tabelę `experiments/thesis_1_table.tex`.*

### 2. Teza 2: Reaktywna Samoadaptacja NSGA-II do Dryfu
```bash
uv run python -m src.main benchmark-thesis-2
```
*Generuje: `charts/thesis_2_drift_adaptation.png/.pdf` oraz tabelę `experiments/thesis_2_table.tex`.*

### 3. Teza 3: Fronty Pareto i Stabilizacja Punktu Kolana (punkt przegięcia)
```bash
uv run python -m src.main benchmark-thesis-3
```
*Generuje: `charts/thesis_3_pareto_tradeoff.png/.pdf` oraz tabelę `experiments/thesis_3_table.tex`.*

### 4. Regeneracja Wszystkich Wykresów z Polską Typografią
```bash
uv run python -m charts.regenerate_all_thesis_charts
```

---

## Podsumowanie Wyników Eksperymentalnych (Rozdział 4)

### Wyniki dla Tezy 1: Redukcja Wymiarowości IPCA
$$\begin{array}{lcccccc}
\hline
\textbf{Wariant} & \textbf{Wymiar ($d$)} & \textbf{Czas online [ms]} & \textbf{Przyspieszenie} & \textbf{Purity [\%]} & \textbf{Silhouette} & \textbf{RAM [MB]} \\
\hline
\text{Pełna przestrzeń SBERT (384d)} & 384 & 26.19 \pm 177.40 & 1.00\times & 19.3\% & \text{N/A} & 567.0 \\
\text{IPCA (d=128)} & 128 & 8.47 \pm 57.27 & 3.09\times & 19.3\% & \text{N/A} & 574.5 \\
\text{IPCA (d=64)} & 64 & 4.16 \pm 27.68 & 6.29\times & 19.3\% & \text{N/A} & 574.8 \\
\text{IPCA (d=32)} & 32 & 12.77 \pm 15.88 & 2.05\times & 56.8\% & 0.032 & 593.2 \\
\mathbf{IPCA (d=16) \star} & \mathbf{16} & \mathbf{3.21 \pm 8.12} & \mathbf{8.17\times} & \mathbf{57.7\%} & \mathbf{0.117} & \mathbf{600.7} \\
\text{IPCA (d=8)} & 8 & 0.95 \pm 5.16 & 27.57\times & 53.9\% & 0.276 & 601.1 \\
\hline
\end{array}$$

### Wyniki dla Tezy 2: Reaktywna Samoadaptacja do Dryfu (modyfikacja parametrów w miejscu vs wymiana w locie)
$$\begin{array}{lccccc}
\hline
\textbf{Model / Strategia} & \textbf{Purity przed dryfem} & \textbf{Stopniowy dryf (Post)} & \textbf{Nagły dryf (Post)} & \textbf{Czas powrotu ($t_{rec}$)} & \textbf{Rekomendowane zastosowanie} \\
\hline
\text{Statyczny (Brak adaptacji)} & 85.8\% & 79.8\% & 88.7\% & \text{Brak powrotu} & \text{Strumienie stacjonarne} \\
\text{Adaptacja modyfikacja parametrów w miejscu} & 85.8\% & 61.5\% & 78.5\% & 150\text{ dokumentów} & \textbf{Stopniowy / lekki dryf} \\
\mathbf{Adaptacja wymiana w locie (NSGA-II) \star} & \mathbf{85.8\%} & \mathbf{72.6\%} & \mathbf{88.7\%} & \mathbf{0\text{ dok. (Natychmiastowy)}} & \textbf{Nagły / głęboki dryf} \\
\hline
\end{array}$$

> [!TIP]
> **Kluczowy Wniosek Teoretyczny dla Komisji:** 
> * **Stopniowy dryf (stopniowy):** Modyfikacja wag w locie (*modyfikacja parametrów w miejscu Parameter Update*) jest optymalna obliczeniowo i zachowuje ciągłość ewolucji mikroklastrów bez konieczności resetowania stanu.
> * **Nagły dryf (nagły):** Wymaga wdrożenia nowego modelu (*trenowany na oknie referencyjnym wymiana w locie*), ponieważ próba gwałtownej zmiany parametrów w locie łamie założenia geometryczne promienia $\epsilon$ i powoduje histerezę przestarzałych mikroklastrów (tzw. *przestarzałe mikroklastry szumy*).

### Wyniki dla Tezy 3: Punkt Kolana (punkt przegięcia) na Frontach Pareto
$$\begin{array}{ccccccc}
\hline
\textbf{Wymiar ($d$)} & \textbf{Rozmiar Frontu} & \textbf{Jakość (Punkt przegięcia)} & \textbf{Złożoność (Punkt przegięcia)} & \mathbf{\epsilon^*} & \mathbf{\mu^*} & \mathbf{\lambda^*} \\
\hline
8 & 30 & 0.2685 & 0.1015 & 0.1403 & 6 & 0.0756 \\
\mathbf{16 \star} & \mathbf{30} & \mathbf{0.0187 - 0.1200} & \mathbf{0.0908} & \mathbf{1.1278} & \mathbf{2} & \mathbf{0.0687} \\
32 & 30 & 0.1732 & 0.1007 & 1.0894 & 2 & 0.0819 \\
64 & 30 & 0.1416 & 0.0927 & 1.1191 & 3 & 0.1500 \\
128 & 30 & 0.1122 & 0.1087 & 0.1920 & 6 & 0.0198 \\
\hline
\end{array}$$

