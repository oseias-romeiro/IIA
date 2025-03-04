

# Função que verifica se a nova rainha não entra em conflito com as rainhas já existentes
def verifica_rainhas(rainhas:list[tuple], nova_rainha:tuple):
    for linha, coluna in rainhas:
        linha_nova, coluna_nova = nova_rainha

        # condições de conflito
        if (
            linha == linha_nova # mesma linha
            or coluna == coluna_nova # mesma coluna
            or abs(linha - linha_nova) == abs(coluna - coluna_nova) # mesma diagonal
        ):
            return False
    return True
