# Note
- Ci va bene fare un resize quadrato perchè le facce sono apprssimativamente un quadrato
- Look up table per le features su similarity.
- Rete from scratch: backbone -> resnet50

## Implementazione rete from scratch
- Resnet50 layer finali:
  ```
  173) conv5_block3_out (Activation)       (None, 7, 7, 2048)     0           ['conv5_block3_add[0][0]']  
  174) avg_pool (GlobalAveragePooling2D    (None, 2048)           0           ['conv5_block3_out[0][0]']
  175) predictions (Dense)                 (None, 1000)           2049000     ['avg_pool[0][0]']
    
  ---
  Total params: 25,636,712
  Trainable params: 25,583,592
  Non-trainable params: 53,120
  ---  
  ```
  Dato che il numero di parametri è già abbastanza alto, il penultimo layer utilizzato è sempre un GlobalAveragePooling2D anzihcè un flatten. Infine vengono aggiunti 2 distinti layer con lo scopo di multitask learning.
- Loss function: somma tra binary cross entropy geenre e mean square error su età. La loss viene pesata in modo diverso tra genere e età.