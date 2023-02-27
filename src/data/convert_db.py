import re
import sqlite3

import sqlite_spellfix

from src.data.utils import untokenize
from src.paths import DB_PATH, PROCESSED_DATA_DIR


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    curs = conn.cursor()
    curs.execute(
        "CREATE TABLE texts(page_id INTEGER, page_name TEXT, og_page_name TEXT, text TEXT)"
    )
    curs.execute(
        "CREATE TABLE lines(page_id INTEGER, line_num INTEGER, line TEXT, line_extra TEXT)"
    )
    curs.execute("CREATE INDEX page_id_index ON texts(page_id)")
    curs.execute("CREATE INDEX page_id_line_num_index ON lines(page_id, line_num)")
    curs.execute("CREATE INDEX page_id_index_2 ON lines(page_id)")
    curs.execute("CREATE INDEX line_num_index ON lines(line_num)")
    conn.close()
    print("Created", db_name)


def transfer_database(old_db, new_db):
    conn_old = sqlite3.connect(old_db)
    curs_old = conn_old.cursor()
    conn_new = sqlite3.connect(new_db)
    curs_new = conn_new.cursor()
    doc_count = curs_old.execute("SELECT Count(*) FROM documents").fetchone()[0]
    query = curs_old.execute("SELECT * FROM documents")
    page_id = 0
    results = query.fetchmany(50000)
    total = len(results)
    print(f"Starting transfer of {doc_count} documents.")
    while len(results) > 0:
        text_rows = []
        line_rows = []
        for result in results:
            page_name, page_body = result
            if page_name == "" or page_body == "":
                continue
            og_page_name = page_name
            page_name = untokenize(page_name, replace_underscore=True)
            page_body = untokenize(page_body)
            # [untokenize(page_line) for page_line in page_lines]
            text_rows.append((page_id, page_name, og_page_name, page_body))
            for page_line in page_body.split("\n"):
                split = page_line.split("\t")
                if not split[0].isdigit():
                    continue
                line_num, line = [int(split[0]), split[1]]
                line_extra = "\t".join([x for x in split[2:]])
                line_rows.append((page_id, line_num, line, line_extra))
            page_id += 1
        curs_new.executemany(
            "INSERT INTO texts(page_id, page_name, og_page_name, text) VALUES (?, ?, ?, ?)",
            text_rows,
        )
        curs_new.executemany(
            "INSERT INTO lines(page_id, line_num, line, line_extra) VALUES (?, ?, ?, ?)",
            line_rows,
        )
        conn_new.commit()
        print(f"{total} of {doc_count} transferred.")
        results = query.fetchmany(50000)
        total += len(results)
    conn_old.close()
    conn_new.close()


import unicodedata


# via scikit-learn
def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart
    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.
    Parameters
    ----------
    s : string
        The string to strip
    See Also
    --------
    strip_accents_ascii : Remove accentuated char for any unicode symbol that
        has a direct ASCII equivalent.
    """
    try:
        # If `s` is ASCII-compatible, then it does not contain any accented
        # characters and we can avoid an expensive list comprehension
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join([c for c in normalized if not unicodedata.combining(c)])


def get_accent_mapping():
    accent_mapping = {}
    for i in range(8000):  # 8000 here is somewhat arbitrary
        char = chr(i)
        unicode_char = repr(char).lower()
        try:
            ascii_char = strip_accents_unicode(unicode_char)
        except Exception:
            continue
        ascii_char = ascii_char.replace("'", "")
        unicode_char = unicode_char.replace("'", "")
        if (
            ascii_char == unicode_char
            or len(ascii_char) != 1
            or ord(ascii_char) < 97
            or ord(ascii_char) > 122
        ):
            continue
        if ascii_char in accent_mapping:
            accent_mapping[ascii_char].append(unicode_char)
        else:
            accent_mapping[ascii_char] = [unicode_char]
    return accent_mapping


def create_spellfix_table(db_file, drop_old=False):
    conn = sqlite3.connect(db_file)
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_spellfix.extension_path())
    curs = conn.cursor()
    if drop_old:
        curs.execute("DROP TABLE IF EXISTS default_cost")
        curs.execute("DROP TABLE IF EXISTS clean_titles_default_cost")
    default_cost_table = (
        "CREATE TABLE default_cost(iLang INT, cFROM TEXT, cTO TEXT, iCOST INT);"
    )
    curs.execute(default_cost_table)
    conn.commit()
    curs.execute(
        "CREATE VIRTUAL TABLE clean_titles_default_cost USING spellfix1(edit_cost_table='default_cost')"
    )
    page_titles = curs.execute("SELECT page_id, page_name FROM texts")
    cleaned_page_titles = []
    for page_id, page_title in page_titles:
        cleaned_title = re.sub(r"\([^)]*\)", "", page_title).strip()
        cleaned_page_titles.append((page_id, cleaned_title))
    curs.executemany(
        "INSERT INTO clean_titles_default_cost(rowid, word) VALUES (?, ?)",
        cleaned_page_titles,
    )
    conn.commit()
    conn.close()


def create_fts4_table(db_file):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()
    curs.execute("CREATE VIRTUAL TABLE clean_titles_fts4 USING fts4")
    page_titles = curs.execute("SELECT page_id, page_name FROM texts")
    cleaned_page_titles = []
    for page_id, page_title in page_titles:
        cleaned_title = re.sub(r"\([^)]*\)", "", page_title).strip()
        cleaned_page_titles.append((page_id, cleaned_title))
    curs.executemany(
        "INSERT INTO clean_titles_fts4(rowid, content) VALUES (?, ?)",
        cleaned_page_titles,
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    initial_db = PROCESSED_DATA_DIR / "init.db"
    if not DB_PATH.is_file():
        create_database(DB_PATH)
    transfer_database(initial_db, DB_PATH)
    create_spellfix_table(DB_PATH, True)
