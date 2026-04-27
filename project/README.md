# ASCII Wireframe Renderer

## Popis

Tento projekt se zabývá perspektivním vykreslováním 3D objektů do terminálu. Dále umožňuje jednoduché ovládání těchto objektů.

## Funkcionality
- Načtení 3D meshů z .obj souborů
- Instanciace objektů (více objektů sdílí jeden mesh)
- Animování objektů (rotace, pohyb a škálování)
- Perspektivní vykreslování do terminálu pomocí braillova fontu nebo pomocí `#` pro kompatibilitu
- Vykreslovaní fungujici i pro hrany překračující `near plane`

## Příklady
- Načtení dvou meshů (krychle a pravidelný dvaceti-stěn)
- Tvorba 3 objektů (jeden dvaceti-stěn a dvě krychle sdílející jeden mesh)
- Ukázka použití animací (rotující dvaceti-stěn, pohybujici se krychle a krychle měnící svůj rozměr)
- Ukázka transformace kamery (rotující kamera)
- Jednoduchý vykreslovací cyklus