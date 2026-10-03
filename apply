r"""
add_logos.py  --  put the CERBERUS logos into the app and the README

Expects, next to app.py:
    static/logo-symbol.svg     symbol only   -> favicon + small header marks
    static/logo-full.svg       full logo     -> README top

Run from the folder that contains app.py:
    python add_logos.py

What it changes
  * every page in templates/ : adds the symbol as the browser-tab icon
  * sessions.html  : the CSS square next to "CERBERUS" becomes the symbol
  * index.html     : the pulsing dot next to "CERBERUS" becomes the symbol
  * review.html    : the symbol appears in front of "Session: ..."
  * login.html, signup.html : the symbol appears above the "CERBERUS" title
    (the full logo's small text would be unreadable at that size, so it is
     used in the README only)
  * README.md      : the "# CERBERUS" title becomes the centered full logo

/static/ files are public in your app (the login page can load them), so the
logo shows before sign-in.  Evidence images under /static/violations/ are still
protected.

Safe by design: any page where an anchor is missing is skipped with a note;
an anchor that matches more than once stops everything before anything is
written.  Backups: <file>.logo.bak.  Running it twice is harmless.
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "templates")
README = os.path.join(HERE, "README.md")

SYMBOL = "{{ url_for('static', filename='logo-symbol.svg') }}"
FAVICON = f'<link rel="icon" type="image/svg+xml" href="{SYMBOL}">'


def img(style, size_attr):
    return (f'<img src="{SYMBOL}" alt="" {size_attr} '
            f'style="{style}">')


HEADER_MARK = img("display:block;height:28px;width:auto", 'height="28"')
TITLE_MARK = img("display:block;height:44px;width:auto;margin:0 0 10px", 'height="44"')
INLINE_MARK = img("display:inline-block;height:26px;width:auto;vertical-align:middle;margin-right:10px",
                  'height="26"')

# page -> list of (name, literal anchor, replacement)
PAGE_EDITS = {
    "sessions.html": [
        ("header mark",
         '<span class="brand-mark" aria-hidden="true"></span>',
         HEADER_MARK),
    ],
    "index.html": [
        ("header mark",
         '<span class="pulse-dot"></span>',
         HEADER_MARK),
    ],
    "review.html": [
        ("title mark",
         '<h1>Session: {{ session_id }}</h1>',
         f'<h1>{INLINE_MARK}Session: {{{{ session_id }}}}</h1>'),
    ],
    "login.html": [
        ("title mark",
         '<h1>CERBERUS</h1>',
         TITLE_MARK + '\n    <h1>CERBERUS</h1>'),
    ],
    "signup.html": [
        ("title mark",
         '<h1>CERBERUS</h1>',
         TITLE_MARK + '\n    <h1>CERBERUS</h1>'),
    ],
}
ALL_PAGES = ["sessions.html", "index.html", "review.html", "login.html",
             "signup.html", "mfa_setup.html"]


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write(path, text):
    shutil.copy2(path, path + ".logo.bak")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def main():
    missing = [n for n in ("logo-symbol.svg", "logo-full.svg")
               if not os.path.isfile(os.path.join(HERE, "static", n))]
    if missing:
        sys.exit("Missing in static/: " + ", ".join(missing) + ". Nothing was changed.")
    if not os.path.isdir(TEMPLATES):
        sys.exit(f"templates/ not found next to this script ({HERE}).")

    work, notes = [], []

    for page in ALL_PAGES:
        path = os.path.join(TEMPLATES, page)
        if not os.path.isfile(path):
            notes.append(f"{page}: not found, skipped")
            continue
        src = read(path)
        out = src
        done = []

        # favicon
        if 'rel="icon"' in out or "rel='icon'" in out:
            pass
        elif out.count("</title>") == 1:
            out = out.replace("</title>", "</title>\n" + FAVICON, 1)
            done.append("favicon")
        else:
            notes.append(f"{page}: no single </title> found, favicon skipped")

        # logo marks
        for name, anchor, replacement in PAGE_EDITS.get(page, []):
            marker = next(m for m in (HEADER_MARK, TITLE_MARK, INLINE_MARK) if m in replacement)
            if marker in src:
                continue   # already patched
            n = out.count(anchor)
            if n > 1:
                sys.exit(f"[{page}] '{name}' anchor matches {n} times (expected 1). Nothing was changed.")
            if n == 0:
                notes.append(f"{page}: '{name}' anchor not found, skipped")
                continue
            out = out.replace(anchor, replacement, 1)
            done.append(name)

        if out != src:
            work.append((path, out))
            notes.append(f"{page}: " + ", ".join(done))
        else:
            notes.append(f"{page}: nothing to change")

    if os.path.isfile(README):
        src = read(README)
        if "logo-full.svg" in src:
            notes.append("README.md: logo already there")
        else:
            block = ('<p align="center">\n'
                     '  <img src="static/logo-full.svg" alt="CERBERUS" width="300">\n'
                     '</p>')
            out, n = re.subn(r"^# CERBERUS[ \t]*\r?$", block, src, count=1, flags=re.M)
            if n == 1:
                work.append((README, out))
                notes.append("README.md: full logo added")
            else:
                notes.append("README.md: '# CERBERUS' title not found, skipped")
    else:
        notes.append("README.md: not found, skipped")

    for path, text in work:
        write(path, text)

    print("\n".join(notes))
    print(f"\n{len(work)} file(s) changed. Restart the app and hard-refresh (Ctrl+F5) "
          "so the browser picks up the new icon.")


if __name__ == "__main__":
    main()