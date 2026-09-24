# evostream

Kod do pracy magisterskiej „Ewolucyjna samoadaptacja potoków NLP: wielokryterialna optymalizacja klasteryzacji strumieniowej tekstu” (Jakub Walkowicz, AGH, 2026).

System grupuje strumień dokumentów tekstowych w czasie rzeczywistym: osadzenia SBERT są rzutowane metodą IPCA, klasteryzowane algorytmem DenStream i grupowane w tematy. Detektor dryfu pojęcia bez użycia etykiet wykrywa zmianę tematów, a NSGA-II dobiera wtedy nowe parametry modelu.

## Wymagania

- Python 3.12 i [uv](https://docs.astral.sh/uv/)
- Docker (Kafka i PostgreSQL – tylko do uruchomienia pełnego systemu)

```bash
uv sync
cp .env.example .env
```

## Uruchomienie systemu

```bash
docker compose up -d kafka-broker-1 db   # Kafka i PostgreSQL
uv run python -m src.main setup          # tematy Kafki i tabele w bazie
uv run python -m src.main daemon         # klasteryzacja strumienia
uv run python -m src.main ingest         # wysyłanie dokumentów do Kafki (osobny terminal)
```

Panel demonstracyjny (Streamlit, bez Kafki i bazy):

```bash
uv run python -m src.main ui
```

## Eksperymenty

Wyniki trafiają do katalogów `results/` obok skryptów.

| Podrozdział pracy | Polecenie |
|---|---|
| 2.3.2 Faza offline | `uv run python -m experiments.denstream_offline_failure.denstream_offline_failure_proof` |
| 3.3 Analiza wymiarowości | `uv run python -m experiments.dimensionality.exp_dimensionality` |
| 3.4 Zakres ε | `uv run python -m experiments.param_bounds.epsilon.exp_epsilon_bounds` |
| 3.4 Zakres λ | `uv run python -m experiments.param_bounds.lambda.exp_lambda_bounds` |
| 4.1 Teza 1 | `uv run python -m experiments.theses.thesis_1.exp_thesis_1_ipca --radius river` |
| 4.1 Formuła promienia | `uv run python -m experiments.denstream_radius.verify_radius`<br>`uv run python -m experiments.denstream_radius.exp_radius_robustness` |
| 4.2 Teza 2 | `uv run python -m experiments.theses.thesis_2.run_sweep`<br>`uv run python -m experiments.theses.thesis_2.summarize_thesis_2` |
| 4.3 Teza 3 | `uv run python -m experiments.theses.thesis_3.exp_thesis_3_pareto` |

Pierwsze uruchomienie pobiera zbiór 20 Newsgroups i oblicza osadzenia SBERT; kolejne korzystają z zapisanych plików.

## Struktura

```
config/config.yaml   parametry systemu
src/domain/          preprocessing, klasteryzacja, detektor dryfu, NSGA-II
src/apps/            demon, generator strumienia, panel Streamlit
src/main.py          polecenia CLI
experiments/         eksperymenty z rozdziałów 3 i 4
charts/              rysunki z rozdziałów 1 i 2
```
