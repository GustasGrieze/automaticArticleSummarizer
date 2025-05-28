import pathlib, sys, textwrap, re, warnings, requests, datetime
from rich import print
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepmultilingualpunctuation import PunctuationModel
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "facebook/bart-large-cnn"
DEVICE     = "cpu"
SUMMARY_LOG = pathlib.Path("santraukos.txt")

print(f"[italic dim]🔄  Kraunamas santraukos modelis {MODEL_NAME} ({DEVICE}) …[/]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

print("[italic dim]🔄  Kraunamas skyrybos modelis (deepmultilingualpunctuation) …[/]")
punct_model = PunctuationModel()

def restore_punct(text: str, chunk_size: int = 400) -> str:
    words = text.split()
    restored_parts = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        restored_parts.append(punct_model.restore_punctuation(chunk))
    return " ".join(restored_parts)

def count_sentences(txt: str) -> int:
    return len(re.findall(r"[.!?]", txt))

def summarize(text: str,
              target_sentences: int = 8,
              max_tokens: int = 240,
              min_tokens: int = 120,
              tries: int = 3) -> str:

    for _ in range(tries):
        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           max_length=1024).to(DEVICE)

        ids = model.generate(
            **inputs,
            max_length=max_tokens,
            min_length=min_tokens,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        summary = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        if count_sentences(summary) >= target_sentences:
            return summary

        max_tokens += 80
        min_tokens += 40

    return summary

TXT = pathlib.Path("tekstynas.txt")
if TXT.exists():
    raw_articles = TXT.read_text(encoding="utf-8").split("===ARTICLE===")[1:]
else:
    raw_articles = []

def parse_title(block: str) -> str:
    for line in block.splitlines():
        if line.startswith("Title: "):
            return line[7:]
    return "Be pavadinimo"

def get_body(block: str) -> str:
    return block.split("===\n", 1)[-1].strip()

titles = [parse_title(b) for b in raw_articles]

def fetch_url(url: str) -> str:
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    txt = "\n".join(paragraphs)

    if len(txt.split()) < 100:
        sys.exit("Nepavyko išgauti pakankamai teksto iš URL.")
    return txt

def log_summary(source: str, summary: str):
    """Prideda santrauką į santraukos.txt (prie galo)."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with SUMMARY_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {source}\n{summary}\n\n")

while True:
    print("\n[bold]Santraukos režimas[/bold]")
    print("[cyan]1[/] – rinktis straipsnį iš tekstyno")
    print("[cyan]2[/] – įvesti straipsnio URL")
    print("[cyan]q[/] – išeiti")

    mode = input("Pasirinkite → ").strip().lower()
    if mode in {"q", ""}:
        print("👋  Iki!")
        break

    if mode == "1":
        if not raw_articles:
            print("⛔️  Tekstynas nerastas arba tuščias.")
            continue
        print("\n[bold]Straipsnių sąrašas[/bold]")
        for i, t in enumerate(titles, 1):
            print(f"[cyan]{i:>2}[/] {t}")
        try:
            idx = int(input("\nĮveskite straipsnio numerį → "))
            body  = get_body(raw_articles[idx - 1])
            label = f"Tekstynas: {titles[idx - 1]}"
        except (ValueError, IndexError):
            print("⛔️  Netinkamas numeris.")
            continue

    elif mode == "2":
        url = input("Įveskite straipsnio URL → ").strip()
        print("[italic dim]🔄  Parsisiunčiamas turinys iš URL …[/]")
        try:
            body = fetch_url(url)
        except Exception as e:
            print(f"⛔️  Klaida: {e}")
            continue
        label = f"URL: {url}"
    else:
        print("⛔️  Neatpažintas pasirinkimas.")
        continue

    print("\n[bold]--- SANTRAUKA ---[/bold]\n")
    body     = restore_punct(body)
    summary  = summarize(body)
    print(textwrap.fill(summary, width=100))
    
    log_summary(label, summary)
    print(f"\n\n[italic dim]💾  Santrauka įrašyta į „{SUMMARY_LOG}“[/]")
