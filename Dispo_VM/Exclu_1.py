from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
from Projects import PROJECTS

engine_source = create_engine("mysql+pymysql://nidec:MaV38f5xsGQp83@162.19.251.55:3306/gestion_commun")
engine_dest   = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator")
DATE_EPOCH = "2025-01-01 00:00:00"

def copy_indispos_to_exclu_ticket():
    try:
        query = f"""
        SELECT 
            id_indispo,
            Projet_global,
            project_id,
            borne_id,
            starttime,
            endtime,
            responsability,
            ticket_id
        FROM indispos
        WHERE deleted_at IS NULL
          AND Projet_global = 'ELTO'
          AND starttime IS NOT NULL
          AND starttime >= '{DATE_EPOCH}'
          AND (
                endtime IS NULL
                OR (endtime >= '{DATE_EPOCH}' AND endtime >= starttime)
              )
        """
        print("Récupération des données depuis gestion_commun.indispos...")
        df = pd.read_sql(text(query), engine_source)
        
        # Filtrer par projets
        df = df[df['project_id'].isin(PROJECTS)]
        print(f"Filtrage project_id selon Projects.py : {len(df)} lignes conservées")
        
        # borne_id NULL -> 'all'
        df['borne_id'] = df['borne_id'].fillna('all')
        print("Transformation borne_id : NULL -> 'all'")
        
        # Gérer les doublons de ticket_id
        df_with_ticket = df[df['ticket_id'].notna()]
        df_without_ticket = df[df['ticket_id'].isna()]
        
        # Supprimer les doublons sur ticket_id 
        df_with_ticket = df_with_ticket.drop_duplicates(subset=['ticket_id'], keep='first')
        
        # Recombiner les DataFrames
        df = pd.concat([df_with_ticket, df_without_ticket], ignore_index=True)
        print(f"Suppression des doublons ticket_id : {len(df)} lignes conservées")
        
        df['starttime'] = df['starttime'].replace({pd.NaT: None})
        df['endtime']   = df['endtime'].replace({pd.NaT: None})
        print("Conversion des dates NULL pour MySQL")
        
        # insertion
        with engine_dest.connect() as conn:
            print("\nVidage de la table Exclu_ticket...")
            conn.execute(text("TRUNCATE TABLE Exclu_ticket"))
            conn.commit()
        
        print("Insertion des données dans Dispo.Exclu_ticket...")
        df.to_sql(
            name='Exclu_ticket',
            con=engine_dest,
            if_exists='append',
            index=False,
            chunksize=1000
        )
        
        print(f"{len(df)} lignes insérées avec succès dans Exclu_ticket")
        
        with engine_dest.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM Exclu_ticket"))
            count = result.fetchone()[0]
            print(f"Vérification : {count} lignes dans Exclu_ticket")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de la copie : {str(e)}")
        return False
        
    finally:
        engine_source.dispose()
        engine_dest.dispose()

if __name__ == "__main__":
    print("=" * 60)
    print("Démarrage de la copie indispos -> Exclu_ticket")
    print("=" * 60)
    print(f"Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    success = copy_indispos_to_exclu_ticket()
    
    print(f"\nFin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    if success:
        print("Copie terminée avec succès !")
    else:
        print("La copie a échoué. Vérifiez les logs ci-dessus.")
    print("=" * 60)