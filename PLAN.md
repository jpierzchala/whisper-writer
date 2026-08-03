# Plan rewampingu WhisperWriter — lipiec 2026

> **Status: zaimplementowane** (branch `revamp/gpt-transcribe-and-model-refresh`).
> Wyniki i korekty założeń — patrz „Co wyszło z testów na żywo" niżej oraz §2.
> Niezrealizowane świadomie: PR 6 (relokacja/wersjonowanie configu) i hot-reload
> bez restartu — opisane na końcu jako pozostały zakres.

## Co wyszło z testów na żywo (2026-07-31)

| Ustalenie | Skutek |
| --- | --- |
| v1 API działa dla LLM na wszystkich hostnames Azure, także Foundry | LLM przeszedł na v1 |
| v1 **nie** obsługuje audio (404 nawet dla działającego `whisper`) | audio zostało na ścieżce legacy — korekta pierwotnego planu |
| `gpt-transcribe-global` działa z `api-version=2025-03-01-preview` | wdrożone jako Twój default |
| `keywords[]` akceptowane (17 i 49 termów) | słownik trafia do STT |
| singularne `language` **cicho ignorowane** przez gpt-transcribe | przełączanie pola per model jest obowiązkowe, nie kosmetyczne |
| A/B cleanup na 5 realnych dyktatach PL: `gpt-5.4-global` 3,69 s vs `gpt-5.6-luna-datazone` **2,75 s** | Luna przyjęta jako default (26% szybciej, ~12× taniej, zero odrzuceń) |
| Luna gubiła łamania linii wymagane przez prompt | złagodzone zaostrzeniem reguły w prompcie — poprawa jest wyraźna, ale niedeterministyczna (patrz „Pozostałe ograniczenia") |
| Polski transkrybowany poprawnie | otwarte pytanie zamknięte |


Plan przygotowany do przekazania do implementacji (Opus). Stan wiedzy: 2026-07-31.
Źródła: analiza kodu (całe `src/` + `tests/`), dokumentacja OpenAI/Microsoft Learn
(GPT Transcribe, model retirements, Azure v1 API), benchmarki latencji Artificial
Analysis (artificialanalysis.ai/models/gpt-5-6-luna i pokrewne), skill
`ai-model-selection` (BI Shared). Plan przeszedł weryfikację adversarialną
(zgodność z kodem + kompletność) — poprawki naniesione.

---

## 0. Diagnoza stanu obecnego (skrót)

- **Transkrypcja** (`src/transcription.py`): 4 providerzy API (openai, azure_openai,
  deepgram, groq — ten ostatni przez SDK Groq) + lokalny faster-whisper/Vosk.
  OpenAI/Azure/Deepgram przez surowe `requests.post` **bez timeoutów**; blok
  WAV-encode skopiowany 4×. Azure używa legacy endpointu
  `/openai/deployments/{d}/audio/transcriptions?api-version=2024-02-01`.
- **LLM post-processing** (`src/llm_processor.py`, 1085 linii): 6 providerów, surowe
  `requests`; routing po prefiksie nazwy modelu (`gpt-5`, `o1`); ścieżki Responses API
  mają `max_output_tokens=1024` hardcoded (ucina długie dyktaty) i timeout=60, ale
  ścieżki chat/Claude/Gemini nie mają ani limitu tokenów, ani timeoutu; provider
  Claude martwy (KeyError na `self.config['endpoint']` — klucz nie istnieje
  w schemacie, wyjątek połykany); `get_available_models` używa usuniętego pre-1.0
  API `openai.Model.list()` (zawsze zwraca `[]`) i złego endpointu Ollamy
  (`/api/models` zamiast `/api/tags`).
- **Lista modeli w UI** (`settings_window.py:405-419`): gpt-5.4, gpt-5.3-chat-latest,
  gpt-5.2, gpt-5.1, gpt-5.1-mini, gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini,
  gpt-3.5-turbo — większość martwa lub umierająca (szczegóły w §4).
- **UI**: PyQt5, okna frameless z `setFixedSize`, settings 800×800 bez możliwości
  zmiany rozmiaru; ~10 hardcodowanych wyjątków od generacji ze schematu; **plik
  `settings_window.py` urwany w połowie funkcji** `toggle_transcription_provider_options`
  (kończy się na linii 1130 przed pętlą `setVisible(True)`) — pola providera
  transkrypcji po przełączeniu chowają się i nigdy nie pokazują z powrotem;
  **przycisk „Reset to saved settings" crashuje** (TypeError — patrz §5.1);
  odświeżanie listy modeli blokuje wątek UI; brak dark mode; restart całej aplikacji
  po każdym zapisie ustawień.
- **Konfiguracja**: schemat YAML dobrze pomyślany, ale wszystkie ścieżki są
  CWD-relative (`src/config.yaml`, `.env`, `assets/` — `run.py` musi być odpalany
  z rootu repo); klucze zapisywane przez UI poza schematem (`*_file_path` ×3,
  `model_options.api.api_key`) — dryf; brak wersjonowania/migracji; mapowanie
  sekretów config↔keyring zduplikowane w 4 łańcuchach if/elif (z czego jeden —
  `load_settings` — to martwy kod).
- **Słownik domenowy** użytkownika żyje dziś wewnątrz `clenup prompt.txt` (49 termów
  kanonicznych + reguły normalizacji homofonów) i działa dopiero na etapie LLM cleanup.

---

## 1. Decyzje architektoniczne i rekomendacje

### 1.1 Rekomendacja modelu do LLM post-processingu: **gpt-5.6-luna**

Zgodnie z metodyką skilla `ai-model-selection` (frontier: Luna none→max → Sol medium+;
Terra tylko długi kontekst; eskalacja effortu przed zmianą klasy modelu) oraz danymi
o latencji (Artificial Analysis, pomiary z lipca 2026 — patrz „Źródła latencji" niżej):

| Tryb | Model | Reasoning effort | Uzasadnienie |
| --- | --- | --- | --- |
| **cleanup** (dyktat) | `gpt-5.6-luna` | **`none`** | Zadanie mechaniczne (klasa „formatowanie/proste edycje" → Luna Low/None). Latencja krytyczna — użytkownik czeka na wklejenie tekstu: TTFT 0,72–0,74 s, 173–284 tok./s → **~2 s end-to-end** dla typowego dyktatu (~300 tok.) vs ~4,3 s dziś na gpt-5.4 (TTFT 0,80 s, 85 tok./s). Jakość: AA Intelligence Index 27 non-reasoning — 6. miejsce na 79 modeli non-reasoning, powyżej gpt-5.2/5.4 non-reasoning. Cena $0,20/$1,20 za 1M — **~12× taniej niż gpt-5.4** ($2,50/$15). Azure GA do 2028-01-11 (najdłuższy runway). |
| **instruction** | `gpt-5.6-luna` | **`low`** (konfigurowalne) | Tryb „wykonaj polecenie" bywa trudniejszy; Luna low dodaje ~0,6 s TTFT. Eskalacja per skill: najpierw effort (medium/high), dopiero potem klasa modelu. Terra — tylko gdyby pojawił się długi kontekst (tu nie występuje); nigdy Sol low. |
| **text cleanup** (zaznaczony tekst) | jak cleanup | `none` | Ten sam charakter zadania. |

Krytyczne szczegóły implementacyjne:

- **GPT-5.6 domyślnie używa effort `medium`** (inaczej niż 5.1, które defaultowało do
  `none`) — effort MUSI być zawsze jawnie wysyłany, inaczej płacimy ~1–2 s cichego
  „myślenia" na każdym dyktacie. Obecny kod wysyła `none` — zachować, ale patrz §3.2
  (effort jako ustawienie).
- `minimal` NIE istnieje na gpt-5.1+ (tylko na oryginalnym gpt-5) — nie używać.
- Predicted outputs nie działają z gpt-5.x — na OpenAI i Azure wspierane tylko dla
  gpt-4o/4o-mini/4.1 (developers.openai.com/api/docs/guides/predicted-outputs,
  learn.microsoft.com/azure/foundry/openai/how-to/predicted-outputs) — nie dodajemy.
- Priority processing na Azure nie obejmuje Luny — lista wspieranych modeli zawiera
  z rodziny 5.6 tylko terra i sol (MS Learn, concepts-priority-processing); Luna jest
  wystarczająco szybka bez niego.
- `verbosity: low` + rozsądny `max_output_tokens` skracają generację.
- Przed przełączeniem defaultu: **baseline A/B na ≥10 reprezentatywnych polskich
  dyktatach** (gpt-5.4 vs luna none) — wymóg skilla (walidacja po zmianie modelu,
  nie „odpowiedział coś"). Brak publicznego benchmarku polskiego cleanupu — trzeba
  zmierzyć u siebie. Uwaga na heurystyki odrzucania wyniku cleanup (§3.9) — mogą
  cicho fałszować A/B.

**Źródła latencji** (do niezależnej weryfikacji przy implementacji; liczby to mediany
AA przy ~1k tokenów promptu, więc dla naszych ~300 tokenów TTFT będzie równy lub
lepszy): artificialanalysis.ai/models/gpt-5-6-luna (+warianty -non-reasoning,
-low, /providers), /gpt-5-4-non-reasoning, /gpt-5-2-non-reasoning;
developers.openai.com/api/docs/guides/latest-model (zalecenie: effort none jako
baseline latencji); learn.microsoft.com/azure/foundry/openai/how-to/latency.

Wymagany nowy deployment na Azure: `gpt-5.6-luna` (GlobalStandard) na zasobie
`NAME` — obok istniejących `gpt-5.4-global`/`gpt-5.2`.

### 1.2 Słownik domenowy: rozbić na osobny byt (rekomendacja: TAK, rozbijamy)

Nowe pojęcie w aplikacji: **Vocabulary (słownik termów)** — jedna lista, wiele
zastosowań:

1. **`keywords[]` w GPT Transcribe** — poprawa rozpoznawania już na etapie STT
   (to hinty, nie wymuszenia: „Keywords are hints, not required output").
2. **Sekcja glosariusza doklejana automatycznie do promptu cleanup** (opcjonalna,
   domyślnie włączona) — LLM nadal wymusza formę kanoniczną i normalizuje homofony.
3. (bez zmian) `find_replace_file` zostaje jako deterministyczna, ostateczna warstwa
   wymuszeń — dla mapowań, które muszą zadziałać zawsze (np. „grubs → Groups").

Projekt:

- Nowe ustawienia w sekcji transkrypcji:
  - `vocabulary_file` (str, ścieżka; **default: `~/.whisperwriter/vocabulary.txt`**,
    spójnie z relokacją configu w §6) — plik tekstowy, jeden term na linię,
    `#` komentarz;
  - `vocabulary_inline` (str, multiline) — alternatywa dla krótkich list; oba źródła
    są scalane;
  - `use_vocabulary_in_transcription` (bool, default true) — wysyłaj jako `keywords[]`
    (tylko modele, które to wspierają — patrz §2);
  - `use_vocabulary_in_cleanup` (bool, default true) — doklej glosariusz do system
    promptu cleanup jako wygenerowaną sekcję („Canonical terms: …").
- Walidacja termów pod API: bez `<`, `>`, CR, LF (API odrzuca CAŁY request);
  trim + dedup; log ostrzeżenia przy odrzuconym termie.
- Migracja treści: wyciąć listę termów z `clenup prompt.txt` do
  `~/.whisperwriter/vocabulary.txt`; **przed modyfikacją zapisać kopię
  `clenup prompt.txt.bak`** (plik jest ręcznie edytowany przez użytkownika i ma
  niezacommitowane zmiany). W promptcie zostają wyłącznie reguły zachowania (styl,
  output rules, anty-injection, generyczna reguła „normalizuj near-misses do termów
  z glosariusza") — bez wypisanej listy. Migracja półautomatyczna: komunikat dla
  użytkownika + propozycja, nie cicha podmiana.

### 1.3 Warstwa API: przejście na oficjalne SDK `openai` (≥2.x) i Azure v1

- Jeden klient `OpenAI(base_url=..., api_key=...)` obsługuje i OpenAI, i Azure:
  dla Azure `base_url = https://{zasób}.openai.azure.com/openai/v1/` **lub**
  `https://{zasób}.services.ai.azure.com/openai/v1/` (oba formaty oficjalnie
  równoważne — endpoint Foundry z ekranu użytkownika działa wprost). `api-version`
  nie jest już wymagane; klasa `AzureOpenAI` przestaje być potrzebna.
- **Normalizacja endpointu musi obsłużyć także `*.cognitiveservices.azure.com`** —
  to jest AKTUALNY endpoint transkrypcji użytkownika (`src/config.yaml:42`);
  bez tego PR 4 zepsuje działającą konfigurację. Zasada: przyjmij dowolny z trzech
  hostname'ów, dolep `/openai/v1/` jeśli brak.
- W v1 pole `model` = **nazwa deploymentu** (nie nazwa modelu) — spójne z obecną
  semantyką per-deployment.
- Ujednolicone timeouty (SDK ma wbudowane retry/backoff), koniec ręcznego parsowania
  odpowiedzi multipart/chat/responses.
- Surowe HTTP zostaje tylko dla Deepgrama (brak sensownego SDK) — z timeoutem.
  Claude przechodzi na SDK `anthropic`, które JUŻ jest zadeklarowane w
  `pyproject.toml:41` a nigdzie nie importowane (§3.6).
- `pyproject.toml`: podbić `openai>=2.51`; przy okazji audyt `anthropic`/`groq`/
  `google-generativeai` (ta ostatnia biblioteka jest deprecated na rzecz `google-genai`).

### 1.4 Abstrakcja providerów (transkrypcja i LLM)

- `TranscriptionProvider.transcribe(audio, sample_rate, hints) -> TranscriptionResult`
  — wspólna baza: WAV-encode raz, keyring lookup, timeout, klasyfikacja błędów
  (retryable vs fatal), retry w warstwie transkrypcji zamiast w `ResultThread`.
- `LLMProvider.process(text, system_message, params) -> LLMResult(ok/text/error)`
  — main.py dostaje informację o błędzie i może pokazać status w StatusWindow zamiast
  cicho wklejać surowy transkrypt.
- **Rejestr capabilities modeli** zamiast prefix-matchingu nazw: per model/deployment
  deklarujemy: API (chat|responses|transcriptions), wspierane efforty, temperature
  tak/nie, structured outputs, parametr językowy (`language` vs `languages[]`),
  wsparcie `keywords`. Rejestr zasila **i backend, i UI** (widoczność temperatury,
  filtrowanie listy modeli per provider) — dziś prefiksy są zduplikowane w
  `llm_processor.py:28` i `settings_window.py:27`.
- **Rodzina modelu deploymentu Azure — jawne ustawienia** (bez zgadywania po nazwie):
  - `model_options.api.azure_openai_model_family` (options: `gpt-transcribe`,
    `gpt-4o-transcribe`, `whisper`; default `whisper`),
  - `llm_post_processing.azure_openai_llm_cleanup_model_family` i
    `..._instruction_model_family` (options: `gpt-5.x-reasoning`, `chat-legacy`;
    default `gpt-5.x-reasoning`).
  - Migracja: jednorazowa inferencja z obecnych nazw deploymentów użytkownika
    (`gpt-transcribe-global` → gpt-transcribe; `gpt-5.4-global`, `gpt-5.2` →
    gpt-5.x-reasoning), potem już tylko jawny wybór w UI (dropdown przy polu
    deploymentu).

---

## 2. Etap A — GPT Transcribe + Azure Foundry (transkrypcja)

Fakty API (potwierdzone w dokumentacji OpenAI/Microsoft, 2026-07-29):

- Model **`gpt-transcribe`** (batch; zalecany przez OpenAI jako domyślny do plików)
  i `gpt-live-transcribe` (wyłącznie Realtime API). Aplikacja nagrywa klip i wysyła
  całość → **implementujemy batch**; realtime poza zakresem (odnotować jako możliwe
  przyszłe rozszerzenie dla trybu continuous).
- Endpoint: `POST /v1/audio/transcriptions` (multipart), przez SDK:
  `client.audio.transcriptions.create(model=..., file=..., prompt=...,
  extra_body={"keywords": [...], "languages": [...]})` — `keywords`/`languages`
  nie są jeszcze first-class kwargs w SDK.
- **`languages[]` (tablica) ZASTĘPUJE `language`** dla gpt-transcribe — nie wolno
  wysłać obu. Dla użytkownika: `["pl"]`, a przy dyktowaniu z angielskimi wtrąceniami
  warto przetestować `["pl", "en"]`.
- `keywords[]`: literalne termy; bez `<`, `>`, CR, LF (inaczej cały request odrzucony);
  brak udokumentowanego limitu liczby; brak opcji bias/strength.
- Odpowiedź JSON: `{"text": ..., "languages": [{"code": ...}]}`. Brak verbose_json/
  timestampów (to nadal domena whisper-1). Max plik 25 MB (WAV 16 kHz mono int16
  ≈ 32 KB/s → ~13 min nagrania; wystarczające, ale dodać guard z czytelnym błędem).
- Cena: $0.0045/min ($0.27/h audio na Azure GlobalStandard); billing per czas, nie tokeny.
- Azure: deployment użytkownika już istnieje — **`gpt-transcribe-global`** na
  `https://NAME.services.ai.azure.com` (GlobalStandard, 10k TPM /
  10 RPM — dla dyktowania OK, odnotować w opisie ustawienia).

Zmiany:

1. **`config_schema.yaml`**: dodać `gpt-transcribe` do opcji `model_options.api.model`;
   odświeżyć opis (obecny mówi „OpenAI only supports whisper-1" — nieprawda).
   Lista modeli w UI filtrowana per provider z rejestru capabilities (§1.4);
   walidacja pary provider/model przy zapisie.
2. **Konfiguracja języków**: `model_options.common.language` (str) zostaje dla
   whisper-1/gpt-4o-transcribe/local. Nowy klucz `model_options.common.languages`
   (str, CSV np. `"pl,en"`, default pusty). Reguła precedencji w rejestrze
   capabilities: model z `languages[]` → użyj `languages` jeśli ustawione, inaczej
   `[language]` (o ile language ≠ auto → wtedy nic, autodetekcja); model z
   `language` → jak dziś. Nigdy nie wysyłać obu pól.
3. **`transcription.py` → provider OpenAI**: mapping parametrów per model
   (z rejestru): dla `gpt-transcribe` → `languages[]`+`prompt`+`keywords[]`+
   `temperature`; dla `whisper-1`/`gpt-4o-transcribe` → jak dziś.
4. **`transcription.py` → provider Azure**: nowy tryb **v1**:
   `POST {endpoint}/openai/v1/audio/transcriptions`, nagłówek `api-key`,
   `model` = nazwa deploymentu, bez `api-version`. Normalizacja endpointu wg §1.3
   (trzy hostname'y, w tym `cognitiveservices.azure.com`!). Tryb sterowany nowym
   ustawieniem **`azure_api_mode`** (options `v1`|`legacy`, default `v1`) — osobno
   dla transkrypcji (`model_options.api.azure_api_mode`) i LLM
   (`llm_post_processing.azure_api_mode`); w trybie `legacy` używamy dzisiejszej
   ścieżki `/openai/deployments/...` i honorujemy `azure_openai_api_version`
   (ignorowane w `v1`). Obecne wartości użytkownika (`2025-03-01-preview` dla
   transkrypcji, `v1` dla LLM) ujęte w migracji §6.
5. **Vocabulary** (§1.2): wpięcie `keywords[]` przy modelach, które wspierają;
   walidacja i logowanie.
6. **Ustawienia domyślne dla użytkownika** (w jego config, krok migracji):
   provider `azure_openai`, deployment `gpt-transcribe-global`,
   `azure_openai_model_family: gpt-transcribe`, endpoint Foundry, `languages: "pl"`.
7. **Smoke test** (skrypt w `tests/` lub `examples/`): próbka WAV po polsku
   z termami domenowymi → transkrypcja przez (a) OpenAI `gpt-transcribe`,
   (b) Azure deployment `gpt-transcribe-global` z `keywords` i bez → porównanie.
   To jednocześnie rozstrzyga otwarte kwestie API (poniżej).

**ROZSTRZYGNIĘTE testami na żywym zasobie `NAME` (2026-07-31):**

| Pytanie | Wynik |
| --- | --- |
| Czy v1 działa dla LLM na endpointcie Foundry? | **TAK** — `gpt-5.4-global` odpowiada przez `/openai/v1/responses` na wszystkich trzech hostnames (`openai.azure.com`, `services.ai.azure.com`, `cognitiveservices.azure.com`). |
| Czy v1 działa dla **audio**? | **NIE** — `/openai/v1/audio/transcriptions` zwraca `DeploymentNotFound` nawet dla deploymentu `whisper`, który na ścieżce legacy odpowiada 200. **Audio musi zostać na `/openai/deployments/{name}/audio/transcriptions?api-version=...`.** To korekta pierwotnego założenia planu. |
| Czy `gpt-transcribe-global` działa? | **TAK**, na ścieżce legacy z `api-version=2025-03-01-preview` (`2024-06-01` też; `preview` → 404). Zwraca nowy kształt odpowiedzi: `{"text":…, "languages":[{"code":…}], "usage":{"type":"duration","seconds":N}}`. |
| Czy Azure przyjmuje `keywords[]`? | **TAK** — 200 dla 3 i 17 termów, bez błędu. |
| Czy `languages[]` działa i ma efekt? | **TAK** — bez hintu model wykrył `zh`, z `languages[]=pl` zwrócił tekst z `languages:[{"code":"pl"}]`. |
| Czy singularne `language` działa dla gpt-transcribe? | **NIE — jest cicho ignorowane** (nie odrzucane!). Dlatego przełączanie pola per model w rejestrze jest obowiązkowe, a nie kosmetyczne. |
| Polski? | **TAK** — realne nagranie PL przetranskrybowane poprawnie (gpt-transcribe dodał nawet przecinek, którego whisper nie postawił). |
| SDK `openai` 2.51 a Azure legacy | **Działa** — `keywords`/`languages` są pełnoprawnymi parametrami `audio.transcriptions.create` i poprawnie serializują się na ścieżce legacy. |

Wniosek architektoniczny: **LLM → v1, audio → legacy.** Ustawienie `azure_api_mode`
jest osobne dla obu sekcji, z różnymi defaultami.

Pozostała otwarta kwestia: limit liczby `keywords` (17 potwierdzone, 49 nietestowane
— walidacja i tak przycina niepoprawne termy).

---

## 3. Etap B — LLM post-processing: modele i parametry

1. **Nowe defaulty schematu**: `cleanup_model: gpt-5.6-luna`,
   `instruction_model: gpt-5.6-luna` (schema + `default_models` w
   `llm_processor.py`); Azure deployment defaults z `gpt-4.1` na `gpt-5.6-luna`.
2. **Reasoning effort jako ustawienia**: `cleanup_reasoning_effort` (default `none`),
   `instruction_reasoning_effort` (default `low`); opcje
   `none|low|medium|high|xhigh` (bez `minimal`; `max` tylko Responses+5.6 — można
   dodać, gdy wybrano 5.6). Zastępuje hardcodowane `_get_preferred_reasoning_effort`
   i specjalny case `gpt-5.3-chat` (model umiera 2026-08-10 — usunąć
   `GPT53_CHAT_PREFIXES`). Fallback 400-retry (parsowanie „Supported values are:")
   **odtworzyć na wyjątkach SDK** — obecna implementacja jest transport-level na
   `requests` i zniknie wraz z nią.
3. **`max_output_tokens`**: zamiast hardcodowanych 1024 — skalowanie od wejścia
   (np. `max(1024, 2× szacowane tokeny wejścia)`) lub jawne ustawienie w konfiguracji;
   to realny bug ucinający długie dyktaty. Ścieżkom chat (dziś bez żadnego limitu)
   dać ten sam mechanizm.
4. **`verbosity: "low"`** dla cleanup (mniej tokenów = szybciej); structured output
   (`cleaned_text` JSON-schema) zostaje bez zmian.
5. **Azure v1 przez SDK** (§1.3): chat + responses jednym klientem; usunięcie
   heurystyki `_azure_supports_structured_outputs` (parsowanie roku z api-version) —
   w trybie `v1` structured outputs zawsze wspierane, w `legacy` decyduje rejestr
   capabilities.
6. **Naprawa providera Claude** (dziś martwy): przejść na SDK `anthropic`
   (już w `pyproject.toml:41`, nigdy nie importowane) albo hardcode
   `https://api.anthropic.com/v1/messages`; default `claude-haiku-4-5` (szybki,
   tani — profil zadania jak Luna) lub `claude-sonnet-5` — **ID zweryfikować przy
   implementacji w aktualnej dokumentacji Anthropic** (nie było przedmiotem
   researchu). Alternatywa: wyciąć providera (decyzja użytkownika; rekomendacja:
   naprawić — przy nowej abstrakcji to mały koszt).
7. **Odświeżyć defaulty pozostałych providerów**: gemini-1.5-flash → aktualny
   odpowiednik z rodziny `gemini-2.x-flash` (sprawdzić bieżące ID przy implementacji),
   Groq `llama-3.1-8b-instant` → zweryfikować dostępność.
8. **`get_available_models`**: przepisać na `client.models.list()` (SDK), dodać
   wsparcie `azure_openai` (lista deploymentów wymaga ARM API — zamiast tego pokazać
   editable combo z ostatnio używanymi nazwami deploymentów), naprawić Ollama
   (`/api/tags`).
9. **Heurystyki odrzucania wyniku cleanup** (`get_cleanup_rejection_reason`,
   stosowane w `main.py` z cichym fallbackiem do surowego transkryptu): gdy
   structured output sparsował się do `cleaned_text` — **ufać wynikowi**, heurystyki
   stosować tylko dla providerów bez structured outputs. Inaczej zmiana modelu
   i doklejenie glosariusza (§1.2) mogą zwiększyć odsetek cichych odrzuceń
   i zafałszować A/B. Do checklisty A/B: „zero cichych fallbacków w przebiegu".
10. **Migracja konfiguracji użytkownika** (gated na wynik A/B): config użytkownika
    nadpisuje defaulty schematu, więc sama zmiana defaultów NIC nie zmieni u niego.
    Po pozytywnym A/B: podmienić w jego config `cleanup_model`/`instruction_model`
    na `gpt-5.6-luna` i oba deploymenty Azure na nowy deployment luny
    (+ `..._model_family` wg §1.4).

---

## 4. Etap C — usunięcie modeli legacy

Stan cyklu życia (OpenAI deprecations + Azure model-retirement-schedule, 2026-07):

| Model z obecnej listy UI | Status | Decyzja |
| --- | --- | --- |
| `gpt-5.3-chat-latest` | **OpenAI shutdown 2026-08-10** (za ~10 dni!); na Azure `*-chat` już Retired | **USUŃ natychmiast** |
| `gpt-3.5-turbo` | OpenAI shutdown 2026-10-23; Azure: dawno retired | **USUŃ** |
| `gpt-4o`, `gpt-4o-mini` | Azure: Deprecated (4o 2024-05-13 retires 2026-10-01); OpenAI: bez daty, ale nie-featured | **USUŃ** z listy (edytowalne combo i tak pozwoli wpisać ręcznie) |
| `gpt-4.1`, `gpt-4.1-mini` | Azure: Deprecated, retire 2027-04-14 | **USUŃ** z listy |
| `gpt-5.1-mini` | **Nie istnieje w cenniku OpenAI** (jest gpt-5-mini i gpt-5.4-mini) — prawdopodobnie błędne ID | **USUŃ** |
| `gpt-5.1` | GA (Azure do 2027-05-15), ale zbędny przy 5.6 | USUŃ z listy |
| `gpt-5.2` | GA (Azure do 2027-06-08) | zostaw (user używa dziś do instruction) |
| `gpt-5.4` | GA (Azure do 2027-09-02) | zostaw (porównanie/regresja per skill) |

**Nowa lista w UI** (`_default_llm_model_choices` → docelowo dane w rejestrze
capabilities / module stałych, nie w UI): `gpt-5.6-luna` (default), `gpt-5.6-terra`,
`gpt-5.6-sol`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.2`. Combo zostaje edytowalne
(własne ID/deploymenty).

Pozostałe czystki:

- `REASONING_MODEL_PREFIXES = ('gpt-5','o1')` → usunąć `o1` (retirement 2026-10-21/23).
  Uwaga: stała jest zduplikowana w **dwóch** miejscach — `llm_processor.py:28`
  i `settings_window.py:27` (steruje widocznością temperatury w UI); docelowo oba
  miejsca zastępuje rejestr capabilities (§1.4).
- Modele transkrypcji: zostawić `whisper-1` (na OpenAI wspierany; **uwaga:
  Azure `whisper` retires 2026-12-15** — odnotować w opisie), `gpt-4o-transcribe`
  (Azure wersja 2025-03-20 retires 2026-10-15 — opis), dodać `gpt-transcribe`
  (rekomendowany default API). Deepgram nova-2/nova-3 i modele Groq — bez zmian
  (poza zakresem; ewentualna weryfikacja przy implementacji).
- Usunąć mapowanie `chatgpt`→`openai`, legacy pole `azure_openai_llm_deployment_name`
  (po migracji konfiguracji), `LEGACY_CLEANUP_RESPONSE_JSON_FIELD`, martwy odczyt
  klucza keyring `whisper` (`settings_window.py:623`), legacy widget `model`
  („Cleanup Model:" QLineEdit — branch nieosiągalny, klucz nie istnieje w schemacie).

---

## 5. Etap D — rewamp UI

Decyzja ramowa: **zostajemy na PyQt5** (migracja na PyQt6 to osobny, niewnoszący
wartości wysiłek — poza zakresem tego rewampingu). Zakres obejmuje SettingsWindow,
StatusWindow, BaseWindow; **MainWindow (launcher 320×180) i tray icon** dostają
tylko kosmetykę spójną z nowym QSS (rozmiary/tytuły/dark mode) w PR 7b.

### 5.1 Bugfixy P0 (można wydzielić jako pierwszy, mały PR)

1. **`toggle_transcription_provider_options` urwane na linii 1130** — pętla „pokaż
   pola wybranego providera" istnieje, ale plik kończy się przed `setVisible(True)`.
   Pola providera transkrypcji nigdy nie są pokazywane z powrotem. Dopisać brakujące
   ciało pętli (docelowo zastąpi to mechanizm z 5.2).
2. **„Reset to saved settings" crashuje**: `update_widgets_from_config`
   (`settings_window.py:634`) robi `for widget, ... in self.iterate_settings():`,
   a `iterate_settings(self, func)` wymaga argumentu i zwraca `None` → TypeError
   przy każdym kliknięciu. Ta sama wada w martwym `load_settings` (`:779,791`).
3. Bug falsy-default: `get_config_value(...) or meta['value']`
   (`settings_window.py:491-494`) — `False/0/''` wyświetlają się jako default
   schematu. Zamienić na sentinel (`if value is not None`).
4. `browse_system_message_file` wczytuje plik i **wyrzuca zawartość** — wstawić
   `content` do text edita (`settings_window.py:976-980`).
5. Odświeżanie modeli robione synchronicznie na wątku UI — użyć istniejącego,
   nigdy nie zainstancjonowanego `ModelRefreshWorker` na QThread; podczas fetchu
   combo disabled + spinner; **nigdy nie wstawiać komunikatów błędów jako pozycji
   combo** (dziś można zapisać „No models available - Check API key" jako model).
6. Timeouty na wszystkich wywołaniach HTTP transkrypcji i LLM, których dziś nie mają
   (`transcription.py:367,440,495`; `llm_processor.py:433,495,675,890`).
7. (drobne, przy okazji) mouseMoveEvent w `base_window.py:139` testuje stałą
   `Qt.LeftButton` zamiast `event.buttons()` — realnie maskowane przez flagę
   `is_dragging`, poprawka porządkowa, nie funkcjonalna.

### 5.2 Architektura ustawień (schema-driven 2.0)

- Rozszerzyć schemat o metadane UI: `label` (czytelna etykieta zamiast
  `key.replace('_',' ').capitalize()` produkującego „Azure openai llm api version:"),
  `group` (sekcja QGroupBox), `secret: true` (+`keyring_name`) — generuje pole
  password i mapowanie keyring z jednego źródła prawdy, `widget` hint, `min/max`
  dla liczb, **`visible_when`** — deklaratywne warunki z predykatami wartości
  (nie tylko równość: widoczność temperatury zależy od *modelu* przez rejestr
  capabilities). Zastępuje **cztery** ręczne mechanizmy: `toggle_api_local_options`,
  `toggle_llm_provider_options`, `toggle_transcription_provider_options`,
  `update_temperature_visibility`.
  **Uwaga na kolejność**: `ConfigManager.load_default_config` (`utils.py:99-105`)
  ignoruje dodatkowe klucze-rodzeństwo obok `value` (bezpieczne), ale KAŻDY dict
  bez `value` traktuje jako poddrzewo ustawień — metadane na poziomie sekcji
  (np. `_version`, definicje grup) wywalą obecne loadery. Metadane schematu i
  przepisanie loaderów muszą wejść w tym samym PR (patrz §7, kolejność 6 → 7a).
- Rejestr wierszy: `{(category, sub, key): SettingRow}` (label+editor+help w jednym
  obiekcie z `get/set/setVisible`) zamiast stringly-typed `findChild` po objectName.
- `QFormLayout` w `QGroupBox` per podsekcja; pola per-provider w `QStackedWidget`
  (przełączanie provider → podmiana strony zamiast chowania wierszy).
- Lista modeli transkrypcji i LLM filtrowana per provider z rejestru capabilities;
  walidacja par provider/model przy zapisie.
- `QSpinBox`/`QDoubleSpinBox` dla int/float (koniec z `ValueError` przy zapisie).
- Wciągnąć do schematu klucze dziś pozaschematowe: `*_file_path` (3 szt.),
  `vocabulary_*`, nowe klucze z §2 i §3; do pruningu §6 dodać też martwy
  `model_options.api.api_key` (drift w configu użytkownika). Usunąć nieosiągalny
  special-case `recording_volume_reduction` (klucz nie istnieje w schemacie —
  czysty dead code, zero migracji).

### 5.3 Wygląd i zachowanie okien

- Okna resizable: `setMinimumSize` zamiast `setFixedSize`; poprawne tytuły okien
  (dziś każdy title bar to hardcodowane „WhisperWriter" — `base_window.py:77`).
- Jeden centralny arkusz QSS + **dark mode** z palety systemowej; usunąć kilkanaście
  rozsianych `setFont('Segoe UI', …)` (17 wywołań w 4 plikach) i hardcodowane kolory.
- Proponowany układ zakładek (czytelniejszy niż surowe kategorie schematu):
  1. **Transcription** — provider, model/deployment (+model family), język(i),
     vocabulary, opcje local;
  2. **Recording** — klawisze, tryby, urządzenie audio;
  3. **Post-processing** — deterministyczne (find/replace, spacje/kropki) + LLM
     (enabled, provider, cleanup/instruction/text-cleanup w podgrupach z modelem,
     effortem i promptem obok siebie);
  4. **Misc** — logi, autostart, status window.
- StatusWindow: pulse przez `QPropertyAnimation` zamiast przepisywania stylesheetu
  co 15 ms; odkryć ukryty hint skrótów (zakomentowane `show()`); ścieżki assetów
  względem `__file__`; wynieść logikę wymuszania stop continuous-API do kontrolera
  w `main.py` (StatusWindow = czysty prezenter).
- Sound-device enum bez otwierania `InputStream` na każdym urządzeniu przy starcie
  okna (użyć `sd.query_devices()` po metadanych, probe lazy).
- **Zapis bez restartu aplikacji**: hot-reload konfiguracji przez sygnał zmiany
  (restart tylko subsystemu key-listenera/modelu lokalnego, gdy trzeba); usunąć
  modal „application will now restart".
- Usunąć dead code: `create_model_selector`, `refresh_thread`, `model_combo`/
  `showEvent`, martwy `load_settings`, import `set_key`; shimy mockowe QT
  (`QT_WIDGETS_ARE_MOCKED` itd.) usuwać dopiero razem z przejściem testów UI na
  pytest-qt (patrz §7 „Testy").

---

## 6. Etap E — fundament konfiguracji

- **Lokalizacja configu użytkownika**: przenieść z `src/config.yaml` do
  `~/.whisperwriter/config.yaml` z automatyczną migracją przy starcie (fallback:
  stara ścieżka). Motywacja: wszystkie ścieżki są dziś CWD-relative (`utils.py:112,
  130,152`, `.env`, `assets/` — w tym **bezpośrednie użycie w `main.py:122`**,
  które też trzeba zmienić), więc aplikacja działa tylko odpalona z rootu repo.
  (Plik NIE jest śledzony w gicie — `.gitignore:136` — więc to nie jest kwestia
  wycieku, tylko przenośności.)
- **Wersjonowanie schematu + rejestr migracji** uruchamiany w
  `ConfigManager.initialize()`; wchłonąć `migrate_azure_key.py` (ma zresztą złą
  domyślną ścieżkę). Migracje z tego planu: relokacja pliku; `azure_api_mode`
  (obecne `azure_openai_api_version` → tryb legacy tam, gdzie ustawione na wartość
  dated, `v1` → tryb v1); inferencja `*_model_family` z nazw deploymentów (§1.4);
  glosariusz → `vocabulary.txt` (§1.2, półautomatyczna); podmiana modeli LLM
  użytkownika po A/B (§3.10); pruning kluczy spoza schematu.
- Zapis **sparse** (tylko delty od defaultów), pruning kluczy nieznanych schematowi,
  nigdy nie zapisywać placeholderów sekretów.
- Ujednolicić ładowanie schematu (dziś dwa cache: `self.schema` i `_schema`)
  i `.env` (tylko python-dotenv albo wcale — keyring jest źródłem sekretów).
- KeyringManager: logować wyjątki backendu zamiast gołego `except: return ''`.

---

## 7. Kolejność wdrożenia (propozycja PR-ów dla Opusa)

| # | Zakres | Zależności |
| --- | --- | --- |
| PR 1 | Bugfixy P0 z §5.1 (w tym urwany plik i crash Reset!) | brak |
| PR 2 | SDK openai ≥2.x + Azure v1 (`azure_api_mode`) + abstrakcja LLMProvider + rejestr capabilities; naprawa Claude (SDK anthropic); `max_output_tokens` skalowane; §3.9 heurystyki | PR 1 |
| PR 3 | Aktualizacja modeli LLM: nowa lista, defaulty luna, ustawienia reasoning effort, czystka legacy (§3, §4); A/B + migracja configu użytkownika (§3.10) | PR 2 |
| PR 4 | TranscriptionProvider + GPT Transcribe (OpenAI i Azure v1/Foundry, `languages`, model family) + smoke test na żywym deploymencie | PR 2 |
| PR 5 | Vocabulary: schemat, walidacja, wpięcie w keywords[] i cleanup prompt; migracja glosariusza z `clenup prompt.txt` (z backupem) | PR 4 |
| PR 6 | Fundament configu (§6): lokalizacja, wersjonowanie, sparse save, secret metadata — **wraz z przepisaniem loaderów schematu** (constraint z §5.2) | PR 2 |
| PR 7 | UI revamp: 7a (SettingRow + visible_when + metadane schematu — wymaga PR 6), 7b (layout+QSS+dark mode+MainWindow/tray kosmetyka), 7c (hot-reload bez restartu) | PR 6 |
| PR 8 | Czystki końcowe: dead code, opisy schematu, README | wszystkie |

**Testy** — istniejący suite ZŁAMIE się w przewidywalnych miejscach; każdy PR musi
zaktualizować swoje:

- PR 2/4: `test_azure_end_to_end.py`, `test_azure_openai_llm.py`,
  `test_azure_openai_llm_integration.py`, `test_azure_openai_config.py` — assertują
  legacy URL-e `/openai/deployments/...` i `api-version=2024-02-01`.
- PR 3: `test_llm_cleanup_safeguards.py:36` (assert `gpt-5.3-chat` → `medium`),
  `test_llm_processor_migration.py` (mapowanie `chatgpt`→`openai`).
- PR 6: `test_azure_key_migration.py` (wchłonięcie migrate_azure_key.py).
- PR 7: `test_azure_ui_integration.py` (hardcodowane listy pól providerów),
  `test_ui_behavior.py` (fixtury gpt-5.1/gpt-4.1). Usunięcie shimów mockowych QT
  = przejście testów UI na pytest-qt — **zabudżetować w PR 7 jako osobne zadanie**,
  to spora część suite'u.

Weryfikacja końcowa (checklist):

- [ ] Dyktat PL przez Azure `gpt-transcribe-global` z keywords — termy domenowe
      poprawnie rozpoznane lub poprawione w cleanup;
- [ ] A/B jakości cleanup: gpt-5.4 (baseline, zapisany przed zmianą) vs
      gpt-5.6-luna none na ≥10 reprezentatywnych dyktatach; zero cichych
      fallbacków heurystyk odrzucania w przebiegu;
- [ ] Pomiar end-to-end latencji przed/po (oczekiwanie: ~2× szybciej na cleanup;
      zmierzyć realnie — liczby AA to mediany na innym profilu promptu);
- [ ] Długi dyktat (>1024 tokenów wyjścia) nie jest ucinany;
- [ ] Przełączanie providerów w ustawieniach pokazuje/chowa właściwe pola;
      „Reset to saved settings" działa;
- [ ] Zapis ustawień bez restartu aplikacji; wartości `False/0/''` round-tripują;
- [ ] Konfiguracja użytkownika (endpoint `cognitiveservices.azure.com`, deployment
      whisper) działa po migracji bez ręcznych poprawek;
- [ ] `pytest` zielony; smoke `run.py` z katalogu innego niż root repo.

---

## 8. Otwarte kwestie / ryzyka

1. **Azure a `keywords[]`** — najważniejsze ryzyko etapu A; rozstrzygnąć live testem
   pierwszego dnia implementacji (fallback opisany w §2).
2. Jakość Luny na polskim cleanup — brak publicznych benchmarków; decyzja o defaultcie
   po własnym A/B (jeśli zawiedzie: `gpt-5.6-luna low`, potem dopiero wyższe klasy).
3. Publiczny spec Azure v1 GA nie zawiera jeszcze `/audio/transcriptions` (preview
   tak) — być może potrzebny nagłówek preview lub tryb `legacy` z
   `api-version=2025-03-01-preview` jako fallback (mechanizm `azure_api_mode`, §2).
4. Rate limit deploymentu `gpt-transcribe-global` (10 RPM) — przy szybkim dyktowaniu
   seryjnym może się odezwać 429; SDK zrobi retry, ewentualnie podnieść limit w Foundry.
5. Realtime (`gpt-live-transcribe`, delay minimal…xhigh) — świadomie poza zakresem;
   naturalny kandydat na kolejną iterację dla trybu continuous.
6. Liczby latencji AA pochodzą z benchmarku o profilu ~1k tokenów promptu — końcowa
   decyzja o modelu opiera się na własnym pomiarze w aplikacji (checklist §7).

---

## Pozostałe ograniczenia

- **Łamania linii w cleanupie nie są deterministyczne.** Zaostrzona reguła w prompcie
  („jedno zdanie na linię, pusta linia tylko przy zmianie tematu") wyraźnie poprawiła
  zachowanie gpt-5.6-luna, ale na krótkich, spójnych tematycznie dyktatach model nadal
  zwraca jeden akapit. Zmierzone na 4 nagraniach: 6/2/0/4 łamań przy ~6/4/4/2 zdaniach.
  Formatowanie tego typu to deterministyczna operacja na tekście — właściwym miejscem
  jest warstwa `post_processing`, nie prompt.
- **Prompty nie są w repozytorium.** Pliki `*.txt` są ignorowane, bo prompt cleanupu
  zawiera osobiste terminy domenowe i styl pracy autora, a aplikacja czyta go ze
  ścieżki w ustawieniach. Wartości domyślne żyją w `config_schema.yaml`.
