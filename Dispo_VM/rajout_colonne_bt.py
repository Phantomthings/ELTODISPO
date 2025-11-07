from sqlalchemy import create_engine, text
from Projects import PROJECTS

engine_dest = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator")

# Si tes tables sont batt_ et batt2_
BATT_SUFFIX = ["", "2"]
SCHEMA = "indicator"

def main():
    with engine_dest.connect() as conn:
        for site_code in PROJECTS:
            safe_code = site_code.replace("-", "_")

            for suffix in BATT_SUFFIX:
                table_name = f"dispo_blocs_batt{suffix}_{safe_code}"
                full_table_name = f"`{SCHEMA}`.`{table_name}`"

                alter_sql = f"""
                ALTER TABLE {full_table_name}
                ADD COLUMN ICPC VARCHAR(255) NULL DEFAULT NULL COLLATE 'utf8mb4_unicode_ci' AFTER cause;
                """

                try:
                    conn.execute(text(alter_sql))
                    print(f"✅ ALTERED: {table_name}")
                except Exception as e:
                    print(f"❌ ERROR on {table_name}: {e}")

        print("✔️ Toutes les tables BT ont été modifiées.")

if __name__ == "__main__":
    main()
