#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import sys

#python vider_tables.py --only ac batt ou pdc

ENGINE_URL = "mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator"
SCHEMA = "indicator"

# --- Choisir quoi vider (commenter/décommenter) ---
INCLUDE_BATT = True   # tables dispo_blocs_batt*_
INCLUDE_AC   = True   # tables dispo_blocs_ac_*
INCLUDE_PDC  = True   # tables dispo_pdc_n*_

# --- Mode démo (n'exécute pas, affiche seulement) ---
DRY_RUN = False

# --- (optionnel) Filtrer par site: exemple "8822_003" (après remplacement '-') ---
SITE_FILTER = None     # ex: "8822_003" ou None pour tous


@contextmanager
def connect_engine(url: str) -> Engine:
    eng = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    try:
        yield eng
    finally:
        eng.dispose()

def fetch_tables(eng: Engine, like_patterns, site_filter=None):
    """
    Récupère la liste des tables du schéma `SCHEMA` qui matchent
    l'un des patterns LIKE fournis. Optionnellement filtre par site (suffixe).
    """
    where_like = " OR ".join([f"table_name LIKE :p{i}" for i, _ in enumerate(like_patterns)])
    sql = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND ({where_like})
    """
    params = {"schema": SCHEMA}
    for i, p in enumerate(like_patterns):
        params[f"p{i}"] = p

    if site_filter:
        pass

    with eng.begin() as conn:
        rows = [r[0] for r in conn.execute(text(sql), params)]
    if site_filter:
        rows = [t for t in rows if t.endswith("_" + site_filter)]
    return sorted(rows)

def truncate_or_delete(eng: Engine, table_fq: str):
    """
    Tente un TRUNCATE (plus rapide). Si interdit (FK/permissions), fallback en DELETE.
    """
    with eng.begin() as conn:
        try:
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
            conn.execute(text(f"TRUNCATE TABLE {table_fq};"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
            return "TRUNCATE"
        except Exception:
            # fallback
            conn.execute(text(f"DELETE FROM {table_fq};"))
            try:
                conn.execute(text(f"ALTER TABLE {table_fq} AUTO_INCREMENT = 1;"))
            except Exception:
                pass
            return "DELETE"

def main():
    global INCLUDE_AC, INCLUDE_PDC, INCLUDE_BATT, DRY_RUN, SITE_FILTER
    args = sys.argv[1:]
    if "--dry-run" in args:
        DRY_RUN = True
    if "--only" in args:
        try:
            idx = args.index("--only")
            val = args[idx + 1].strip().lower()
            INCLUDE_AC = INCLUDE_PDC = INCLUDE_BATT = False
            if val == "ac":
                INCLUDE_AC = True
            elif val == "pdc":
                INCLUDE_PDC = True
            elif val == "batt":
                INCLUDE_BATT = True
            else:
                print("Valeur de --only inconnue (ac|pdc|batt).")
                return
        except Exception:
            print("Usage: --only ac|pdc|batt")
            return
    if "--site" in args:
        try:
            SITE_FILTER = args[args.index("--site") + 1]
        except Exception:
            print("Usage: --site 8822_003")
            return

    patterns = []
    if INCLUDE_BATT:
        patterns += [r"dispo_blocs_batt\_%", r"dispo_blocs_batt2\_%"]
    if INCLUDE_AC:
        patterns += [r"dispo_blocs_ac\_%"]
    if INCLUDE_PDC:
        patterns += [r"dispo_pdc_n%"]  

    if not patterns:
        print("Aucune catégorie sélectionnée (AC/PDC/BATT). Rien à faire.")
        return

    print(f"Schema: {SCHEMA}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Filtre site: {SITE_FILTER or '(aucun)'}")
    print("Catégories:", ", ".join(
        [name for ok, name in [(INCLUDE_BATT, "BATT"), (INCLUDE_AC, "AC"), (INCLUDE_PDC, "PDC")] if ok]
    ))

    with connect_engine(ENGINE_URL) as eng:
        tables = fetch_tables(eng, patterns, SITE_FILTER)
        if not tables:
            print("Aucune table trouvée pour les motifs et le filtre donnés.")
            return

        print(f"{len(tables)} table(s) ciblée(s) :")
        for t in tables:
            print(f" - {SCHEMA}.{t}")

        if DRY_RUN:
            print("\nDRY_RUN activé : aucune modification effectuée.")
            return

        ok, fail = 0, 0
        for t in tables:
            table_fq = f"`{SCHEMA}`.`{t}`"
            try:
                mode = truncate_or_delete(eng, table_fq)
                print(f"✅ {mode:<8} {table_fq}")
                ok += 1
            except Exception as e:
                print(f"❌ Échec {table_fq} : {e}")
                fail += 1

        print(f"Succès: {ok} | Échecs: {fail}")
        print("Terminé.")

if __name__ == "__main__":
    main()
