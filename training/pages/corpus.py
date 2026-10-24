"""Text for generated pages: prose from the Project Gutenberg books of data/texts.yaml, and invented names, companies, addresses,
IBANs and amounts for forms, invoices and receipts.

Prose comes from the `train` books for the train page split and from the `heldout` books for the validation split. The two
books the page evaluation sets are set in are not in texts.yaml (see data/texts.py), so they cannot be drawn here.

Everything invented is invented: company names are built from syllables, people from common given names and surnames in random
combination, street numbers and phone numbers from random digits, e-mail addresses on the reserved example domains. IBANs have
the right length and valid check digits for their country but random bank and account numbers.
"""

from __future__ import annotations

import os
import random
import re
from datetime import date, timedelta
from pathlib import Path

import yaml

from data import texts as textlib

DATA = Path(__file__).resolve().parent.parent / "data"
LANG_WEIGHT = {"en": 30, "fr": 9, "de": 9, "es": 7, "it": 6, "pt": 4, "nl": 4, "pl": 3, "sv": 2, "da": 2, "fi": 2, "hu": 2, "cs": 2,
               "ro": 1, "ca": 1, "el": 8, "ru": 6, "bg": 3, "sr": 2}  # fmt: skip


_MOJIBAKE = re.compile(r"[ÃÂ][\u0080-\u00bf]|â€|\ufffd|[\u0080-\u009f]")


class TextBank:
    def __init__(self, split: str):
        want = "train" if split == "train" else "heldout"
        books = yaml.safe_load((DATA / "texts.yaml").read_text(encoding="utf-8"))["books"]
        self.by_lang: dict[str, list[dict]] = {}
        for b in books:
            if b["split"] == want:
                self.by_lang.setdefault(b["language"], []).append(b)
        self.root = Path(os.environ["ONEOCR_TEXT_ROOT"])
        self._cache: dict[int, list[str]] = {}
        self.langs = [lg for lg in LANG_WEIGHT if lg in self.by_lang]
        self.used: set[int] = set()  # book ids drawn since the last clear, recorded in the page truth

    def pick_lang(self, rng: random.Random, allowed: set[str] | None = None) -> str:
        langs = [lg for lg in self.langs if allowed is None or lg in allowed]
        return rng.choices(langs, [LANG_WEIGHT[lg] for lg in langs])[0]

    def _paras(self, rng: random.Random, lang: str) -> list[str]:
        b = rng.choice(self.by_lang[lang])
        self.used.add(b["id"])
        if b["id"] not in self._cache:
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            raw = (self.root / f"pg{b['id']}.txt").read_bytes()
            # some editions carry double-encoded text ("Ã¶" for "ö"); '=' marks bold in the plain-text editions
            ps = [p.replace("=", "") for p in textlib.paragraphs(textlib.body_of(raw)) if p.count(" ") > 6 and not _MOJIBAKE.search(p)]
            self._cache[b["id"]] = ps[:4000]
        return self._cache[b["id"]]

    def paragraph(self, rng: random.Random, lang: str, words: tuple[int, int] = (30, 110)) -> str:
        """A run of whole sentences from one book paragraph, within a word-count range where the source allows."""
        ps = self._paras(rng, lang)
        lo, hi = words
        for _ in range(8):
            p = rng.choice(ps)
            ws = p.split()
            if len(ws) < 12:
                continue
            n = min(len(ws), rng.randint(lo, hi))
            start = 0
            cut = ws[start : start + n]
            # end at a sentence end when there is one in the second half
            ends = [i for i, w in enumerate(cut) if w[-1] in ".!?…»”\"'" and i >= n // 2]
            if ends:
                cut = cut[: ends[-1] + 1]
            return " ".join(cut)
        return " ".join(rng.choice(ps).split()[:hi])

    def sentence(self, rng: random.Random, lang: str, words: tuple[int, int] = (2, 8)) -> str:
        """A short phrase for headings and captions: leading words of a paragraph with trailing punctuation removed."""
        p = self.paragraph(rng, lang, (15, 40)).split()
        k = rng.randint(*words)
        s = rng.randint(0, max(0, len(p) - k))
        out = " ".join(p[s : s + k]).strip(" ,;:.—–-“”\"'()«»„‘’")
        return out or "Untitled"


# --- invented content ---------------------------------------------------------------------------------------------------

_SYL1 = "Bran Cor Del Fen Gar Hal Ker Lund Mar Nor Ost Pel Quin Rav Sol Tar Ulm Var Wen Zel Alt Ber Dun Eld Fol Gra Hav Iron Jan".split()
_SYL2 = "wick ford ton berg mont field haven dale stone ridge port gate well mark vale holm brook ley more wood".split()
_BIZ = {
    "en": ["Ltd", "Inc", "LLC", "Group", "& Sons", "Partners", "Supplies", "Logistics", "Systems", "Studio"],
    "de": ["GmbH", "AG", "KG", "& Söhne", "Handels GmbH", "Technik GmbH", "Logistik AG", "Werke"],
    "fr": ["SARL", "SAS", "& Fils", "Services", "Industrie", "Distribution", "Atelier"],
    "es": ["S.L.", "S.A.", "e Hijos", "Servicios", "Distribuciones", "Suministros"],
    "it": ["S.r.l.", "S.p.A.", "& Figli", "Servizi", "Forniture", "Distribuzione"],
    "nl": ["B.V.", "N.V.", "& Zonen", "Diensten", "Groothandel", "Techniek"],
}
_GIVEN = (
    "Anna Ben Clara David Elena Felix Greta Hugo Irene Jonas Katja Luca Marta Nils Olga Pablo Rita Simon Tessa Viktor Wanda "
    "Yusuf Zoe Andrei Beatrice Carlos Dora Emil Frida Gustav Helena Ivan Julia Karl Lena Mateo Nora Oskar Petra Rosa Stefan"
).split()
_FAMILY = (
    "Adler Becker Conti Dvorak Engel Fischer Garcia Hansen Ivanov Jansen Keller Lang Moreau Novak Olsen Pereira Rossi "
    "Schmidt Torres Unger Vogel Weber Xavier Young Zimmer Bauer Costa Dubois Eriksen Fontaine Greco Horvat Jovanovic"
).split()
_STREET = {
    "en": ("{n} {w} Street", "{n} {w} Road", "{n} {w} Avenue", "{n} {w} Lane"),
    "de": ("{w}straße {n}", "{w}weg {n}", "{w}platz {n}", "Am {w}berg {n}"),
    "fr": ("{n} rue {w}", "{n} avenue {w}", "{n} boulevard {w}", "{n} chemin {w}"),
    "es": ("Calle {w} {n}", "Avenida {w} {n}", "Paseo {w} {n}"),
    "it": ("Via {w} {n}", "Corso {w} {n}", "Piazza {w} {n}"),
    "nl": ("{w}straat {n}", "{w}laan {n}", "{w}weg {n}"),
}
_CITY_END = {"en": "ton ford bury field ham", "de": "burg stadt dorf hausen heim", "fr": "ville mont sur-Loire court", "es": "villa mar ejo ada",
             "it": "ano ello ara ona", "nl": "dam hoven wijk recht"}  # fmt: skip
COUNTRY = {"en": ("GB", "United Kingdom"), "de": ("DE", "Deutschland"), "fr": ("FR", "France"), "es": ("ES", "España"),
           "it": ("IT", "Italia"), "nl": ("NL", "Nederland")}  # fmt: skip
LOCALES = tuple(COUNTRY)
_ITEMS = {
    "en": "chair desk lamp notebook pencil folder binder cable adapter monitor keyboard mouse printer toner paper shelf cabinet".split(),
    "de": "Stuhl Schreibtisch Lampe Notizbuch Bleistift Ordner Kabel Adapter Monitor Tastatur Maus Drucker Toner Papier Regal Schrank".split(),
    "fr": "chaise bureau lampe cahier crayon classeur câble adaptateur écran clavier souris imprimante toner papier étagère armoire".split(),
    "es": "silla escritorio lámpara cuaderno lápiz carpeta cable adaptador monitor teclado ratón impresora tóner papel estante armario".split(),
    "it": "sedia scrivania lampada quaderno matita cartella cavo adattatore monitor tastiera mouse stampante toner carta scaffale armadio".split(),
    "nl": "stoel bureau lamp notitieboek potlood map kabel adapter monitor toetsenbord muis printer toner papier plank kast".split(),
}
_SERVICES = {
    "en": ["Consulting hours", "Installation", "Maintenance contract", "Delivery", "Software licence", "Training day", "Support"],
    "de": ["Beratungsstunden", "Installation", "Wartungsvertrag", "Lieferung", "Softwarelizenz", "Schulungstag", "Support"],
    "fr": [
        "Heures de conseil",
        "Installation",
        "Contrat de maintenance",
        "Livraison",
        "Licence logicielle",
        "Journée de formation",
        "Assistance",
    ],
    "es": [
        "Horas de consultoría",
        "Instalación",
        "Contrato de mantenimiento",
        "Entrega",
        "Licencia de software",
        "Día de formación",
        "Soporte",
    ],
    "it": [
        "Ore di consulenza",
        "Installazione",
        "Contratto di manutenzione",
        "Consegna",
        "Licenza software",
        "Giornata di formazione",
        "Supporto",
    ],
    "nl": ["Adviesuren", "Installatie", "Onderhoudscontract", "Levering", "Softwarelicentie", "Trainingsdag", "Ondersteuning"],
}
_FOOD = {
    "en": "coffee tea sandwich salad soup croissant juice water cake apple bread milk cheese pasta rice chicken".split(),
    "de": "Kaffee Tee Brötchen Salat Suppe Croissant Saft Wasser Kuchen Apfel Brot Milch Käse Nudeln Reis Hähnchen".split(),
    "fr": "café thé sandwich salade soupe croissant jus eau gâteau pomme pain lait fromage pâtes riz poulet".split(),
    "es": "café té bocadillo ensalada sopa croissant zumo agua tarta manzana pan leche queso pasta arroz pollo".split(),
    "it": "caffè tè panino insalata zuppa cornetto succo acqua torta mela pane latte formaggio pasta riso pollo".split(),
    "nl": "koffie thee broodje salade soep croissant sap water taart appel brood melk kaas pasta rijst kip".split(),
}
_SPEC = ["A4", "500 g", "1 l", "Typ 3", "2024", "Gr. M", "No. 7", "12 mm", "XL", "Set", "Pro", "Basic", "Plus", "mk II", "3x"]
# country -> (leading letters, digits) of the BBAN, and the total IBAN length
_IBAN_FMT = {"DE": (0, 18, 22), "GB": (4, 14, 22), "FR": (0, 23, 27), "ES": (0, 20, 24), "IT": (1, 22, 27), "NL": (4, 10, 18),
             "AT": (0, 16, 20), "BE": (0, 12, 16), "PL": (0, 24, 28)}  # fmt: skip


def person(rng: random.Random) -> str:
    return f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}"


def company(rng: random.Random, loc: str) -> str:
    base = rng.choice(_SYL1) + rng.choice(_SYL2)
    if rng.random() < 0.35:
        base = f"{base.capitalize()} {rng.choice(_FAMILY)}"
    return f"{base} {rng.choice(_BIZ[loc])}"


def city(rng: random.Random, loc: str) -> str:
    return rng.choice(_SYL1) + rng.choice(_CITY_END[loc].split())


def postcode(rng: random.Random, loc: str) -> str:
    if loc == "en":
        return f"{rng.choice('ABCDEFGHJKLMNPRSTUWY')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.randint(1, 19)} {rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
    if loc == "nl":
        return f"{rng.randint(1000, 9999)} {rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}"
    return f"{rng.randint(10000, 99999)}"


def address(rng: random.Random, loc: str) -> dict:
    street = rng.choice(_STREET[loc]).format(n=rng.randint(1, 240), w=rng.choice(_SYL1) + rng.choice(_SYL2))
    return {"street": street, "postcode": postcode(rng, loc), "city": city(rng, loc), "country": COUNTRY[loc][1], "cc": COUNTRY[loc][0]}


def phone(rng: random.Random, loc: str) -> str:
    code = {"en": "+44", "de": "+49", "fr": "+33", "es": "+34", "it": "+39", "nl": "+31"}[loc]
    return f"{code} {rng.randint(20, 99)} {rng.randint(1000, 9999)} {rng.randint(100, 999)}"


def email(rng: random.Random, who: str) -> str:
    return who.lower().replace(" ", ".").replace("&", "").replace("..", ".").strip(".") + "@example." + rng.choice(["com", "org", "net"])


def iban(rng: random.Random, cc: str) -> str:
    """Correct length and ISO 7064 mod-97 check digits for the country; bank and account numbers are random."""
    letters, digits, total = _IBAN_FMT[cc]
    bban = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(letters)) + "".join(rng.choice("0123456789") for _ in range(digits))
    num = "".join(str(int(ch, 36)) for ch in bban + cc + "00")
    out = f"{cc}{98 - int(num) % 97:02d}{bban}"
    assert len(out) == total and iban_ok(out)
    return out


def iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    return int("".join(str(int(ch, 36)) for ch in s[4:] + s[:4])) % 97 == 1


def group4(s: str) -> str:
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4))


def bic(rng: random.Random, cc: str) -> str:
    return (
        "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(4))
        + cc
        + rng.choice(["XX", "2X", "FF", "MM"])
        + (rng.choice(["", "XXX", "100"]))
    )


def vat_id(rng: random.Random, cc: str) -> str:
    cc = "GB" if cc == "GB" else cc
    return cc + "".join(rng.choice("0123456789") for _ in range({"DE": 9, "GB": 9, "FR": 11, "ES": 9, "IT": 11, "NL": 9}.get(cc, 9)))


def item_name(rng: random.Random, loc: str, kind: str = "goods") -> str:
    if kind == "service":
        return rng.choice(_SERVICES[loc])
    if kind == "food":
        return rng.choice(_FOOD[loc]).capitalize() if rng.random() < 0.7 else rng.choice(_FOOD[loc])
    n = rng.choice(_ITEMS[loc])
    return f"{n.capitalize()} {rng.choice(_SPEC)}" if rng.random() < 0.8 else n.capitalize()


# --- number and date formats --------------------------------------------------------------------------------------------


def money(cents: int, loc: str, group: bool = True) -> str:
    neg = cents < 0
    cents = abs(cents)
    whole, frac = divmod(cents, 100)
    w = f"{whole:,}"
    if loc == "en":
        s = f"{w if group else whole}.{frac:02d}"
    elif loc in ("de", "es", "it", "nl"):
        s = f"{w.replace(',', '.') if group else whole},{frac:02d}"
    else:  # fr: space-grouped
        s = f"{w.replace(',', ' ') if group else whole},{frac:02d}"
    return ("-" if neg else "") + s


def fmt_date(d: date, loc: str, style: int) -> str:
    if loc == "en":
        return [d.strftime("%d/%m/%Y"), d.strftime("%Y-%m-%d"), f"{d.day} {d.strftime('%B %Y')}", d.strftime("%d.%m.%y")][style % 4]
    if loc == "de":
        return [d.strftime("%d.%m.%Y"), d.strftime("%Y-%m-%d"), d.strftime("%d.%m.%y")][style % 3]
    return [d.strftime("%d/%m/%Y"), d.strftime("%Y-%m-%d"), d.strftime("%d.%m.%Y"), d.strftime("%d-%m-%Y")][style % 4]


def rand_date(rng: random.Random, lo: int = 2019, hi: int = 2026) -> date:
    a = date(lo, 1, 1)
    return a + timedelta(days=rng.randint(0, (date(hi, 12, 31) - a).days))
