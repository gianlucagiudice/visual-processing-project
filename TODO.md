A partire dal dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ , che raccoglie una serie di immagini dei volti di attori/celebrità e le relative informazioni riguardanti genere ed età, il gruppo definisce un progetto su tutto l’archivio o su un suo sottoinsieme opportunamente scelto.

Il progetto consiste nella progettazione, realizzazione e valutazione di un sistema e/o app che usi il DB di immagini selezionato. In particolare:
- Viene acquisita la foto di uno o più individui, di cui sia ben visibile il volto (foto non di profilo o di spalle e con sufficiente illuminazione). Tecniche di image enhancement vengono applicate per compensare le condizioni non ideali di acquisizione delle immagini.
- L’immagine viene inviata al server usando il bot Telegram fornito dai docenti. 
- Il volto degli individui viene individuato e opportunamente ritagliato dall’immagine originale. Se nessun volto viene individuato, la fotografia viene automaticamente rifiutata. Diverse tecniche di detection verranno valutate (es. Cascade, Yolo,).
- A partire dai volti individuati viene riconosciuto il genere e stimata l’età. Verranno valutati sia approcci con features handcrafted, sia tecniche di deep learning.
- Vengono inoltre reperiti e visualizzati i volti presenti nel dataset più simili a quelli dell’immagine di input, in modo da indicare il personaggio famoso più simile all’individuo nella foto di input.
- Le performance del sistema vengono analizzate su un opportuno set di immagini campione

Suddivisione dei task

- Face detection usando Cascade e Yolo, con annesse misure di performance. INPUT: l’immagine; OUTPUT: immagine croppata del volto.
  N.B.: le img croppate da wikipedia non sono ben croppate, però hanno un 40% di margine -> possiamo toglierlo. Altrimenti prendiamo il dataset di IMDB
- Task di regressione età tramite CNN (problema: from scratch o finetuned?). INPUT: immagine croppata del volto; OUTPUT: un range di età. (multitasking?) 
  Prima provare con LBP, poi con rete from scratch, poi magari con rete pretrained
- Task di classificazione genere tramite CNN (problema: from scratch o finetuned?). INPUT: immagine croppata del volto; OUTPUT: predizione binaria. (multitasking?)
- Similarity con il di celebrities (può anche essere un db ridotto). INPUT: immagine croppata del volto; OUTPUT: top n immagini più simili. (feature di basso livello della cnn?)
- Merge dei vari moduli (LINK AL BOT: https://github.com/dros1986/python_bot).
- Aggiunta di eventuali task di image enhancement all’inizio della pipeline.
