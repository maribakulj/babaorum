"""E12 : banc de segmentation de pages de presse à colonnes, de bout en bout.

- :mod:`axel.layout.gt` : vérité PAGE (régions, lignes, ordre de lecture) ;
- :mod:`axel.layout.geom` : polygones rastérisés, IoU, lignes de base ;
- :mod:`axel.layout.match` : appariement des lignes et des blocs (D-211) ;
- :mod:`axel.layout.order` : ordre de lecture des blocs (coupe XY) ;
- :mod:`axel.layout.pagetext` : CER de page ordonné, insensible à l'ordre,
  et ventilation des erreurs par cause (D-212).
"""
