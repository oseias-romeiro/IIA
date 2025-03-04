
# Dados de entrada (x1, x2) e saída desejada
entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
saidas_desejadas = [0, 0, 0, 1]

# Constante usada na função de transferência
C = 0.5

def rna_porta_and():
    """
    Implementa uma Rede Neural Artificial simples que simula o comportamento de uma porta AND.
    """
    w1, w2 = 0.5, 0.5 # Pesos iniciais

    print("Treinamento da RNA para simular uma porta AND")
    print(f"Pesos iniciais: w1={w1}, w2={w2}\n")

    # Processo de treinamento
    for epoca in range(1, 11):  # Número máximo de épocas
        print(f"Época {epoca}:")
        erros = 0  # Contador de erros para verificar convergência

        for (x1, x2), y_desejada in zip(entradas, saidas_desejadas):
            # Calcula a soma ponderada
            soma = x1 * w1 + x2 * w2

            # Aplica a função de transferência
            saida_obtida = 1 if soma > C else 0

            # Calcula o erro
            erro = y_desejada - saida_obtida

            # Ajusta os pesos se houver erro
            if erro != 0:
                erros += 1
                # Ajuste dos pesos
                w1 += C * x1 * erro
                w2 += C * x2 * erro

            print(f"  Entrada: ({x1}, {x2}), Soma: {soma}, Saída Obtida: {saida_obtida}, "
                  f"Saída Desejada: {y_desejada}, Erro: {erro}, Pesos: w1={w1}, w2={w2}")

        # Verifica se convergiu
        if erros == 0:
            print("\nTreinamento concluído com sucesso!")
            break
        print("\n")

    # Resultado final
    print(f"Pesos finais: w1={w1}, w2={w2}\n")
    print("Testando RNA treinada...\n")

    # Testa a RNA treinada
    for (x1, x2), y_desejada in zip(entradas, saidas_desejadas):
        soma = x1 * w1 + x2 * w2
        saida_obtida = 1 if soma > C else 0
        print(f"Entrada: ({x1}, {x2}), Saída Obtida: {saida_obtida}, Saída Desejada: {y_desejada}")


# Chamada da função
rna_porta_and()
