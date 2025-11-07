#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from Projects import PROJECTS  # doit contenir la liste de codes sites

ENGINE_URL = "mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator"
SCHEMA = "indicator"

BATT_SUFFIX = ["", "2"]
PDC_NUMS = [1, 2, 3, 4, 5, 6]

#  TEMPLATES DDL 

DDL_BATT = """
CREATE TABLE IF NOT EXISTS `{schema}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `site` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `equipement_id` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `type_equipement` ENUM('BATT') NOT NULL DEFAULT 'BATT' COLLATE 'utf8mb4_unicode_ci',
  `date_debut` DATETIME NOT NULL,
  `date_fin` DATETIME NOT NULL,
  `est_disponible` TINYINT(1) NOT NULL COMMENT '1 = disponible, 0 = indisponible',
  `cause` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `ICPC` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `raw_point_count` INT UNSIGNED NULL DEFAULT '0',
  `processed_at` TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  `batch_id` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `hash_signature` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `duration_minutes` INT GENERATED ALWAYS AS (TIMESTAMPDIFF(MINUTE, `date_debut`, `date_fin`)) STORED,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uq_batt_equip_interval` (`equipement_id`, `date_debut`, `date_fin`) USING BTREE,
  INDEX `idx_batt_equip_debut` (`equipement_id`, `date_debut`) USING BTREE,
  INDEX `idx_batt_site_debut_fin` (`site`, `date_debut`, `date_fin`) USING BTREE
)
COMMENT='Blocs consolidés BATT (DC1, DC2) issus d\\'Influx'
ENGINE=InnoDB
COLLATE='utf8mb4_unicode_ci';
"""

DDL_AC = """
CREATE TABLE IF NOT EXISTS `{schema}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `site` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `equipement_id` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `type_equipement` ENUM('AC') NOT NULL DEFAULT 'AC' COLLATE 'utf8mb4_unicode_ci',
  `date_debut` DATETIME NOT NULL,
  `date_fin` DATETIME NOT NULL,
  `est_disponible` TINYINT(1) NOT NULL COMMENT '1 = disponible, 0 = indisponible',
  `cause` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `ICPC` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `raw_point_count` INT UNSIGNED NULL DEFAULT '0',
  `processed_at` TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  `batch_id` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `hash_signature` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `duration_minutes` INT GENERATED ALWAYS AS (TIMESTAMPDIFF(MINUTE, `date_debut`, `date_fin`)) STORED,
  `split_from_id` INT NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uq_ac_equip_interval` (`equipement_id`, `date_debut`, `date_fin`) USING BTREE,
  INDEX `idx_ac_equip_debut` (`equipement_id`, `date_debut`) USING BTREE,
  INDEX `idx_ac_site_debut_fin` (`site`, `date_debut`, `date_fin`) USING BTREE,
  INDEX `idx_split_from` (`split_from_id`) USING BTREE
)
COMMENT='Blocs consolidés AC issus d\\'Influx'
ENGINE=InnoDB
COLLATE='utf8mb4_unicode_ci';
"""

DDL_PDC = """
CREATE TABLE IF NOT EXISTS `{schema}`.`{table}` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `site` VARCHAR(100) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `pdc_id` VARCHAR(8) NOT NULL COLLATE 'utf8mb4_unicode_ci',
  `type_label` ENUM('PDC') NOT NULL DEFAULT 'PDC' COLLATE 'utf8mb4_unicode_ci',
  `date_debut` DATETIME NOT NULL,
  `date_fin` DATETIME NOT NULL,
  `est_disponible` TINYINT(1) NULL DEFAULT NULL,
  `cause` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `ICPC` VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `raw_point_count` INT UNSIGNED NULL DEFAULT '0',
  `processed_at` TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  `batch_id` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `hash_signature` VARCHAR(64) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci',
  `duration_minutes` INT GENERATED ALWAYS AS (TIMESTAMPDIFF(MINUTE, `date_debut`, `date_fin`)) STORED,
  `split_from_id` INT NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uq_interval` (`date_debut`, `date_fin`) USING BTREE,
  INDEX `idx_debut` (`date_debut`) USING BTREE,
  INDEX `idx_site_debut_fin` (`site`, `date_debut`, `date_fin`) USING BTREE,
  INDEX `idx_split_from` (`split_from_id`) USING BTREE
)
COMMENT='Blocs consolidés PDCn issus d\\'Influx'
ENGINE=InnoDB
COLLATE='utf8mb4_unicode_ci';
"""

#  UTIL 

@contextmanager
def connect_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    try:
        yield engine
    finally:
        engine.dispose()

def run_sql(engine: Engine, sql: str):
    with engine.begin() as conn:
        conn.execute(text(sql))

def sanitize(code: str) -> str:
    # remplace '-' par '_' pour nom de table
    return code.replace("-", "_")

#  MAIN 

def main():
    created_ok = 0
    created_err = 0
    with connect_engine(ENGINE_URL) as eng:
        # Assure le schema par défaut
        run_sql(eng, f"CREATE DATABASE IF NOT EXISTS `{SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")

        for site_code in PROJECTS:
            safe_code = sanitize(site_code)

            # --- BATT x 2 suffixes
            for suffix in BATT_SUFFIX:
                table_batt = f"dispo_blocs_batt{suffix}_{safe_code}"
                try:
                    run_sql(eng, DDL_BATT.format(schema=SCHEMA, table=table_batt))
                    print(f"✅ BATT créé/présent : {table_batt}")
                    created_ok += 1
                except Exception as e:
                    print(f"❌ BATT échec {table_batt} : {e}")
                    created_err += 1

            # --- AC unique
            table_ac = f"dispo_blocs_ac_{safe_code}"
            try:
                run_sql(eng, DDL_AC.format(schema=SCHEMA, table=table_ac))
                print(f"✅ AC créé/présent : {table_ac}")
                created_ok += 1
            except Exception as e:
                print(f"❌ AC échec {table_ac} : {e}")
                created_err += 1

            # --- PDC n1..n6
            for n in PDC_NUMS:
                table_pdc = f"dispo_pdc_n{n}_{safe_code}"
                try:
                    run_sql(eng, DDL_PDC.format(schema=SCHEMA, table=table_pdc))
                    print(f"✅ PDC créé/présent : {table_pdc}")
                    created_ok += 1
                except Exception as e:
                    print(f"❌ PDC échec {table_pdc} : {e}")
                    created_err += 1

    print(f"Créations OK : {created_ok} | Échecs : {created_err}")
    print("Terminé.")


if __name__ == "__main__":
    main()
