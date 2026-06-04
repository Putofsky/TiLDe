import pandas as pd


def build_component_dict(
    df: str,
    field_col: str,
    component_col: str,
    sort_col: str | None = None,
    reset_index: bool = True,
) -> dict:
    """
    Construit un dictionnaire de composants à partir d'un fichier CSV.

    imputs :
        df : DataFrame ou chemin vers un fichier CSV
        field_col : nom de la colonne contenant les id de champ
        component_col : nom de la colonne contenant les id de composant
        sort_col : nom de la colonne utilisée pour trier les données
        reset_index :réinitialise les index des DataFrames
    """

    df = pd.read_csv(df)  # Convertit en DataFrame si ce n'est pas déjà le cas

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame.")

    required_cols = [field_col, component_col]
    if sort_col is not None:
        required_cols.append(sort_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    data = df.copy()

    if sort_col is not None:
        data = data.sort_values([field_col, component_col, sort_col])
    else:
        data = data.sort_values([field_col, component_col])

    result = {}

    for field_id, field_df in data.groupby(field_col, sort=False):
        result[field_id] = {}

        for component_id, component_df in field_df.groupby(component_col,
                                                            sort=False):
            if reset_index:
                component_df = component_df.reset_index(drop=True)

            result[field_id][component_id] = component_df.copy()

    return result
