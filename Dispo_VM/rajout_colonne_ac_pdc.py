from sqlalchemy import create_engine, text
from Projects import PROJECTS

# Connexion MySQL
engine_dest = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator")

PDC_NUMS = [1, 2, 3, 4, 5, 6]
SCHEMA = "indicator"

def main():
    with engine_dest.connect() as conn:
        for site_code in PROJECTS:
            safe_code = site_code.replace("-", "_")

            for pdc in PDC_NUMS:
                table_name = f"dispo_pdc_n{pdc}_{safe_code}"
                full_table_name = f"`{SCHEMA}`.`{table_name}`"

                alter_sql = f"""
                ALTER TABLE {full_table_name}
                ADD COLUMN ICPC VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci' AFTER cause,
                ADD COLUMN split_from_id INT DEFAULT NULL,
                ADD INDEX idx_split_from (split_from_id);
                """

                try:
                    conn.execute(text(alter_sql))
                    print(f"✅ ALTERED: {table_name}")
                except Exception as e:
                    print(f"❌ ERROR on {table_name}: {e}")

            table_ac = f"dispo_blocs_ac_{safe_code}"
            full_table_ac = f"`{SCHEMA}`.`{table_ac}`"
            alter_ac_sql = f"""
            ALTER TABLE {full_table_ac}
            ADD COLUMN ICPC VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci' AFTER cause,
            ADD COLUMN split_from_id INT DEFAULT NULL,
            ADD INDEX idx_split_from (split_from_id);
            """

            try:
                conn.execute(text(alter_ac_sql))
                print(f"✅ ALTERED AC: {table_ac}")
            except Exception as e:
                print(f"❌ ERROR on {table_ac}: {e}")

        print("✔️ Toutes les tables PDC et AC ont été modifiées.")

if __name__ == "__main__":
    main()
