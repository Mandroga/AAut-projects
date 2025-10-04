# %%
%run data_imports.py

# %%
'''
Exercise recognition

O paciente esta sentado

E1 - Brushing hair - Passar a mao na cabeça - mãos 
E2 - Brushing teeth - Passar a mao na boca - mãos 
E5 - Hip flexion - Levanta os joelhos ao peito

O AVC afeta o paciente num dos lados =>
E1, E2 - Pode ou nao conseguir levantar a mao
E3 - Levanta sempre pelo menos um dos joelhos => O desvio padrao dos joelhos é maior que os outros exercicios

Ideia:
- Comparar a altura e o desvio padrao do que deve estar a mexer, com keypoints perto da zona de interesse~
(lavar os dentes - mao, nariz)
Logistic classification
Linear 
SVM
Forest classifier
Neural classifiers
'''