# Modelo de curtailment

Subprojeto experimental para preparar sequências de previsão de constrained-off
eólico por estado.

O código fica em [curtailment-eólico](curtailment-e%C3%B3lico/Readme.md) e
possui duas etapas:

1. download de constrained-off, carga e intercâmbio do ONS;
2. agregação horária, engenharia de atributos e criação de sequências de 168 h.

Esse fluxo é independente do CNN-LSTM de previsão de geração em `ml/`. Execute
os comandos a partir da pasta `curtailment-model`, conforme o README interno.
