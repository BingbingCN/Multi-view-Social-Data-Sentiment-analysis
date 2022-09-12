# Multi-view-Social-Data-Sentiment-analysis
`Dataset.ipynb`

This is a unimodal (text) sentiment analysis project.We compared the performance of different optimizers when training Bi-LSTM, CNN, the following is the change image of accuracy and loss：
## BiLSTM

![IY0G )JVJ7~Y1S4$LUQ58_W](https://user-images.githubusercontent.com/94735262/189563370-6fc807f2-6ded-4316-a2a7-a69d60c37dbf.png)

We found that SGD with different learning rates can not perform well when training Bi-LSTM. When using AdamW and Adam, the convergence speed of loss is the fastest.
## CNN

![@ZKG}K%Q8H0ZLR DV}$%{I](https://user-images.githubusercontent.com/94735262/189563504-76c73559-7220-4674-8a98-b7c42516da0f.png)

Similar to Bi-LSTM, Adam's performance is very good, and AdamW's loss may fluctuate at some times, and the loss can converge when using SGD，but the convergence rate is slow
