# TODO

## Intro
A partire dal dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ , che raccoglie una serie di immagini dei volti di attori/celebrità e le relative informazioni riguardanti genere ed età, il gruppo definisce un progetto su tutto l’archivio o su un suo sottoinsieme opportunamente scelto.

Il progetto consiste nella progettazione, realizzazione e valutazione di un sistema e/o app che usi il DB di immagini selezionato. In particolare:
- Viene acquisita la foto di uno o più individui, di cui sia ben visibile il volto (foto non di profilo o di spalle e con sufficiente illuminazione). Tecniche di image enhancement vengono applicate per compensare le condizioni non ideali di acquisizione delle immagini.
- L’immagine viene inviata al server usando il bot Telegram fornito dai docenti. 
- Il volto degli individui viene individuato e opportunamente ritagliato dall’immagine originale. Se nessun volto viene individuato, la fotografia viene automaticamente rifiutata. Diverse tecniche di detection verranno valutate (es. Cascade, Yolo,).
- A partire dai volti individuati viene riconosciuto il genere e stimata l’età. Verranno valutati sia approcci con features handcrafted, sia tecniche di deep learning.
- Vengono inoltre reperiti e visualizzati i volti presenti nel dataset più simili a quelli dell’immagine di input, in modo da indicare il personaggio famoso più simile all’individuo nella foto di input.
- Le performance del sistema vengono analizzate su un opportuno set di immagini campione

## Tasks
Suddivisione dei task:
1. Enhancement immagine
   
   1. Equalizzazione istogramma (?)
2. Face detection usando Cascade e Yolo, con annesse misure di performance. INPUT: l’immagine; OUTPUT: immagine croppata del volto.
3. Task di regressione età tramite CNN (problema: from scratch o finetuned?). INPUT: immagine croppata del volto; OUTPUT: un range di età. (multitasking?) 
4. Task di classificazione genere tramite CNN (problema: from scratch o finetuned?). INPUT: immagine croppata del volto; OUTPUT: predizione binaria. (multitasking?)

5. Similarity con il di celebrities (può anche essere un db ridotto). INPUT: immagine croppata del volto; OUTPUT: top n immagini più simili. (feature di basso livello della cnn?)
   
   1. Filtrare per genere e poi età (filtraggio iterativo)
   2. Considerare foto retrieval indicizzate con struttura ad albero.
6. Merge dei vari moduli (LINK AL BOT: https://github.com/dros1986/python_bot).
7. Scraping retrieval

## Note di implementazione
- Look up table per le features su similarity.

## Classfiicazione
1. Rete from scratch
2. VGG pretreinata su imagenet
3. Handcrafted


## Detection
1. Cascade
2. Yolo

## Divisione lavoro
- Enhancement: Lucrezia
- Detection: 
  - Cascade: Luca
  - Yolo: Gianluca
- Classificazioone:
  - Rete scratch: Gianluca
    - Resnet a caso
  - VGG pretreinata: Luca
  - Handcrafted: Lucrezia
    - Divisione della faccia in 4
      - LBP
      - Colore
      - HAAR
      - SIFT
