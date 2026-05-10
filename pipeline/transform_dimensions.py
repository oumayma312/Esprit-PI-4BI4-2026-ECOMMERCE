from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _replace_text_series(s: pd.Series, old: str, new: str) -> pd.Series:
    if s is None:
        return s
    # PowerQuery ReplaceText is substring replacement, case-sensitive.
    return s.astype("string").str.replace(old, new, regex=False)


def transform_dim_supplier(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PowerQuery replacements on datawarehouse.DimSupplier.

    Notes:
      - Sort steps are applied for determinism only; replacements don't depend on sort.
    """

    if df is None or df.empty:
        return df

    out = df.copy()
    if "supplier" not in out.columns:
        return out

    out = out.sort_values(by=["supplier"], kind="mergesort")

    replacements = [
        ("2.11 STONEWARE", "2M STONEWARE"),
        ("3P", "3P PRINT-PRODUCTION-FIXELS"),
        ("Aramex", "Aramex Tunisie"),
        ("Aramex Tunisie Tunisie", "Aramex Tunisie"),
        ("Batinconfort Mestiri", "Baticonfort Mestiri"),
        ("Zoub ceramic", "ZOUBA THE CERAMIC STUDIO"),
        ("Vivo Energy (Shell)", "VIVO ENERGY"),
        ("Shell (Vivo Energy)", "VIVO ENERGY"),
        ("St� Laouiti Artisanat", "Sté Laouiti Artisanat"),
        ("Bilel Feki", "Bilel ElFEKI"),
        ("EL FEKI BILEL", "Bilel ElFEKI"),
        ("Foire International Sousse", "Foire Internationale de Sousse"),
        ("Oc�an bougies", "OCEAN BOUGIES"),
        ("ENNAIM", "ENNAIM Céramique"),
        ("Karima Dachraoui", "KERAMOS"),
        ("Sonia Zouba", "ZOUBA THE CERAMIC STUDIO"),
        ("ENNAIM Céramique Céramique", "ENNAIM Céramique"),
        ("HIDE AWAY", "Hide Away Tunisia"),
        ("Hamila stoneware", "2M STONEWARE"),
        ("Riadh Louaiti", "Riadh Laouiti"),
        ("Quincaillerie Zormani Hammadi", "Quincaillerie Zormani"),
        ("Quincaillerie Generale", "Quincaillerie Générale"),
        ("OXAHOST SARL", "OXA HOST SARL"),
        ("Oriental Design Kammoun", "ORIENTAL DESIGN"),
        ("LA QUINCAILLERIE DU SAHEL", "Quincaillerie Du Sahel"),
        ("Helmi Abbassi", "HELMI ABASSI"),
        ("Laouiti Artisanat", "Sté Laouiti Artisanat"),
        ("Kamel Hidri", "Kamel HYDRY"),
        ("Karim Balleh", "Karim Balah"),
        ("L'avenir", "STE AVENIR"),
        ("MHN", "MHN lustres en nattes"),
        ("MHN lustres en nattes lustres en nattes", "MHN lustres en nattes"),
        ("Myfitness_Megrine", "MIM Fitness Megrine"),
        ("Orange Tunisie", "ORANGE"),
        ("Sté Sté Laouiti Artisanat", "Sté Laouiti Artisanat"),
    ]

    for old, new in replacements:
        out["supplier"] = _replace_text_series(out["supplier"], old, new)

    out = out.sort_values(by=["supplier"], kind="mergesort")
    return out


def transform_dim_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PowerQuery transformations on datawarehouse.DimCustomer."""

    if df is None or df.empty:
        return df

    out = df.copy()
    if "customer" not in out.columns:
        return out

    out = out.sort_values(by=["customer"], kind="mergesort")

    replacements = [
        (
            "Attijari Bank , centre d'affaires Régions Tunis",
            "Attijari Bank , centres d'affaires Régions Tunis",
        ),
        (
            "Centres d'affaires Centre & Cap bon",
            "Attijari Bank , centres d'affaires Régions Tunis",
        ),
        ("Digital Services VAL", "Digital Services Value"),
        ("Tunisia Call Center", "Tunis Call Center"),
        ("WIFAK BANK", "WIFAK INTERNATIONAL BANK"),
        ("Wifak BAN", "WIFAK INTERNATIONAL BANK"),
        ("Sofia TEC", "SOFIA TECH"),
    ]

    for old, new in replacements:
        out["customer"] = _replace_text_series(out["customer"], old, new)

    # Add column type
    if "channelFK" in out.columns:
        out["type"] = out["channelFK"].map(lambda x: "B2B" if x == 1 else "B2C")
    else:
        out["type"] = "B2C"

    # Distinct on customer (keep first after sort)
    out = out.drop_duplicates(subset=["customer"], keep="first")

    out["customer"] = _replace_text_series(out["customer"], "Force Management TRA", "Force Management")

    out = out.sort_values(by=["customer"], kind="mergesort")

    # Replace channelFK values 4 -> 1
    if "channelFK" in out.columns:
        out.loc[out["channelFK"] == 4, "channelFK"] = 1

    # Add customer_group
    cust = out["customer"].astype("string")
    out["customer_group"] = cust.where(~cust.str.contains("Attijari Bank", na=False), "Attijari Bank")

    return out


def transform_dim_product(df: pd.DataFrame) -> pd.DataFrame:
    """Add column `produit` to DimProduct based on product name rules (PowerQuery equivalent)."""

    if df is None or df.empty:
        return df

    out = df.copy()
    if "product" not in out.columns:
        return out

    product = out["product"].astype("string")
    p = product.str.lower()

    result = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")

    def apply_when(mask: pd.Series, label: str) -> None:
        nonlocal result
        mask = mask.fillna(False)
        result = result.mask(result.isna() & mask, label)

    def has(sub: str) -> pd.Series:
        return p.str.contains(sub, na=False, regex=False)

    # Order matters (else-if chain)
    apply_when(has("bonbonniere") | has("bonbonnière") | has("bonbonniére") | has("bomboniere"), "Bonbonnière")
    apply_when(has("bouteille"), "Bouteille en Céramique")
    apply_when(has("bannière"), "Bannière")
    apply_when(has("carafe"), "Carafe")
    apply_when(has("cruche"), "Cruche en Céramique")
    apply_when(has("serveitte"), "Porte serviettes en Céramique")
    apply_when(has("poste"), "Postes")
    apply_when(has("bougie"), "Bougies")
    apply_when(has("bol"), "Bol Zouba")
    apply_when(has("mug"), "Mug et Plateau en Céramique")
    apply_when(has("verre"), "Set de 6 Verres")
    apply_when(has("action"), "Action Vérité Bitounsi")
    apply_when(has("tala3"), "TALA3 BITOUNSI")
    apply_when(has("bekri"), "Jeu Nes Bekri Kalou")
    apply_when(has("vase") | has("vasl"), "Vase")
    apply_when(has("assiette"), "Assiette")
    apply_when(has("mathred"), "Mathred")
    apply_when(has("boîte") | has("boite"), "Boîte en Céramique")
    apply_when(has("plateau"), "Plateau")
    apply_when(has("planche"), "Planche Céramique")
    apply_when(has("lunch"), "Lunch Box")
    apply_when(has("baume"), "Baume à Lèvre")
    apply_when(has("boudha") | has("bouddha"), "Boudha en Béton")
    apply_when(has("veilleuse"), "Veilleuse")
    apply_when(has("poche"), "Vide-poches")
    apply_when(has("zazoua") | has("zazwa"), "Zazwa en Cuivre")
    apply_when(has("séa"), "Set « Séa »")
    apply_when(has("photophore"), "Photophores Artisanaux")
    apply_when(has("tote"), "Tote Bag")
    apply_when(has("flamme"), "Coffret Flamme d’artisan")
    apply_when(has("bougeoir"), "Bougeoirs Ethniques en Céramique")
    apply_when(has("tasse") | has("tassa"), "Sous Tasse artisanale en Céramique")
    apply_when(has("tortue"), "Tortue en béton")
    apply_when(has("tanit"), "Tanit en Céramique")
    apply_when(has("stylo"), "Stylo LilasDream")
    apply_when(has("plat"), "Sous Plat en Céramique")
    apply_when(has("soupiere") | has("soupiére"), "Soupiére")
    apply_when(has("gourmet"), "Set Gourmet L’Éveil")
    apply_when(has("vaisselle"), "Set de Vaisselle Organica")
    apply_when(has("service"), "Service de Table")
    apply_when(has("jeu"), "Jeu d'Ambiance")
    apply_when(has("bouddha") | has("boudha"), "Bouddha")
    apply_when(has("fouta"), "Fouta Artisanale en Coton")
    apply_when(has("couffin"), "Couffin artisanal en fibre végétale")
    apply_when(has("déodorant"), "Déodorant Solide")
    apply_when(has("diffuseur") | has("diffiseur"), "Diffuseur Voiture")
    apply_when(has("ensemble"), "Ensemble décoratif en résine")
    apply_when(has("olivéa"), "Set Olivéa en Grès")
    apply_when(has("séli"), "Set Séli en Grès Noirs")
    apply_when(has("mots"), "Set Mots du Cœur")
    apply_when(has("support"), "Support PC en Bois")
    apply_when(has("sculpture"), "Sculpture")
    apply_when(has("poisson"), "Poisson décoratif artisanal en céramique")
    apply_when(has("coupe"), "Coupes à vin")
    apply_when(has("broc") | has("broche"), "Broche Cuivre Artisanale")
    apply_when(has("coffret"), "Coffret")
    apply_when(
        has("échecs")
        | has("echequier")
        | has("échequier")
        | has("echéquier")
        | has("échiquier"),
        "Échiquier en Bois d'Olivier",
    )
    apply_when(has("moia"), "Set Moia Béton")
    apply_when(has("mouton"), "Mouton Décoratif en Laine")

    out["produit"] = result.fillna(product)
    return out
