import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from anastruct.fem.system import SystemElements

st.set_page_config(page_title="MRD Scanner Pro", layout="wide")

st.title("📐 Scanner Estrutural Hiperestático (MRD + OpenCV)")
st.write("Insira a foto do caderno. O sistema lerá os nós (amarelo) e gerará a estrutura para o cálculo.")

# --- SIDEBAR: ENTRADAS DE MATERIAIS ---
with st.sidebar:
    st.header("1. Propriedades Mecânicas")
    arquivo_imagem = st.file_uploader("Suba a foto do sistema:", type=["png", "jpg", "jpeg"])
    
    st.markdown("---")
    E_gpa = st.number_input("Módulo de Elasticidade E (GPa)", value=210)
    A_cm2 = st.number_input("Área da Seção A (cm²)", value=50)
    I_cm4 = st.number_input("Momento de Inércia I (cm⁴)", value=2000)
    
    # Conversões para Sistema Internacional (kN, m)
    E_val = E_gpa * 1e6
    A_val = A_cm2 / 10000
    I_val = I_cm4 / 1e8

# --- DETECÇÃO REAL DE CORES E GEOMETRIA ---
if arquivo_imagem is not None:
    # Converter arquivo enviado para imagem OpenCV
    file_bytes = np.asarray(bytearray(arquivo_imagem.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    
    # Máscara para Nós (Amarelo)
    yellow_lower = np.array([20, 80, 80])
    yellow_upper = np.array([35, 255, 255])
    mask_nos = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Encontrar os nós amarelos na foto
    contours_nos, _ = cv2.findContours(mask_nos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lista_nos = []
    
    for idx, cnt in enumerate(contours_nos):
        if cv2.contourArea(cnt) > 30:  # Filtrar pequenos ruídos
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m20"] / M["m00"])
                lista_nos.append((cx, cy))
                
    # Ordenar nós da esquerda para a direita (para facilitar a lógica do formulário)
    lista_nos = sorted(lista_nos, key=lambda x: x[0])
    num_nos = len(lista_nos)
    
    # Montar arranjo preliminar de barras ligando os nós em sequência física
    # (Ex: se achou 3 nós ordenados por X, assume Barra 1: Nó 1->2, Barra 2: Nó 2->3)
    barras_detectadas = []
    if num_nos > 1:
        for i in range(num_nos - 1):
            barras_detectadas.append((f"Nó {i+1}", f"Nó {i+2}"))

    # --- CORPO DA INTERFACE ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Elementos Mapeados na Foto")
        
        # Desenhar círculos nos nós detectados para dar feedback visual ao usuário
        img_feedback = img.copy()
        for i, (cx, cy) in enumerate(lista_nos):
            cv2.circle(img_feedback, (cx, cy), 15, (0, 255, 255), -1) # Círculo amarelo preenchido
            cv2.putText(img_feedback, f"No {i+1}", (cx - 20, cy - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        st.image(img_feedback, channels="BGR", use_column_width=True)
        st.success(f"O OpenCV identificou: {num_nos} Nós e {len(barras_detectadas)} Barra(s).")

    with col2:
        st.subheader("⚙️ Calibração Manual Baseada no Desenho")
        
        if num_nos < 2:
            st.error("Desenhe pelo menos 2 nós amarelos para que o sistema reconheça uma barra.")
        else:
            comprimentos = {}
            st.markdown("#### Comprimento das Barras (m)")
            for i, barra in enumerate(barras_detectadas):
                comprimentos[i] = st.number_input(
                    f"Comprimento da {barra[0]} até o {barra[1]}:", 
                    value=4.0 if i==0 else 3.0, 
                    min_value=0.1, 
                    key=f"len_{i}"
                )
            
            st.markdown("---")
            st.markdown("#### Configuração de Cargas e Apoios")
            no_carga = st.selectbox("Selecione o Nó onde a Força Verde atua:", [f"Nó {i+1}" for i in range(num_nos)], index=1 if num_nos>1 else 0)
            val_fx = st.number_input(f"Força Horizontal Fx no {no_carga} (kN):", value=10.0)
            val_fy = st.number_input(f"Força Vertical Fy no {no_carga} (kN):", value=-20.0)

    # --- MOTOR DE CÁLCULO (ANASTUCT) ---
    if num_nos >= 2:
        st.markdown("---")
        if st.button("🚀 Resolver Estrutura via Rigidez Direta"):
            try:
                ss = SystemElements()
                
                # Construir vetor de coordenadas reais baseado nos comprimentos manuais
                # O Nó 1 começa em (0,0). Os seguintes acumulam a distância projetada.
                coords_reais = {1: [0.0, 0.0]}
                
                # Lógica de projeção simples: se a variação no pixel Y for muito maior que X na foto, a barra é vertical
                for i in range(len(barras_detectadas)):
                    p1_pix = lista_nos[i]
                    p2_pix = lista_nos[i+1]
                    dx_pix = abs(p2_pix[0] - p1_pix[0])
                    dy_pix = abs(p2_pix[1] - p1_pix[1])
                    
                    L = comprimentos[i]
                    x_anterior, y_anterior = coords_reais[i+1]
                    
                    if dy_pix > dx_pix: # Barra majoritariamente Vertical na foto
                        # Se no pixel o Y cresce para baixo, tratamos a orientação geométrica real para cima
                        coords_reais[i+2] = [x_anterior, y_anterior + L]
                    else: # Barra majoritariamente Horizontal na foto
                        coords_reais[i+2] = [x_anterior + L, y_anterior]

                # Adicionar os elementos estruturais ao anastruct
                for i in range(len(barras_detectadas)):
                    ss.add_element(location=[coords_reais[i+1], coords_reais[i+2]], EA=E_val*A_val, EI=E_val*I_val)
                
                # Condições de Contorno automáticas baseadas em convenção (Ex: Engaste na base Nó 1, Apoio Simples no último nó)
                ss.add_support_fixed(node_id=1)
                if num_nos > 2:
                    ss.add_support_hinged(node_id=num_nos)
                
                # Aplicar a carga no nó escolhido
                idx_no_carga = int(no_carga.split(" ")[1])
                ss.add_node_load(node_id=idx_no_carga, Fx=val_fx, Fy=val_fy)
                
                # Solucionar o sistema K * U = F
                ss.solve()
                
                # --- APRESENTAÇÃO DOS RESULTADOS ---
                st.subheader("📊 Diagramas de Esforços Solicitantes")
                g1, g2, g3 = st.columns(3)
                
                with g1:
                    st.write("**Momento Fletor (M)**")
                    fig_m, ax_m = plt.subplots()
                    ss.plot_bending_moment(show=False, ax=ax_m)
                    st.pyplot(fig_m)
                with g2:
                    st.write("**Esforço Cortante (V)**")
                    fig_v, ax_v = plt.subplots()
                    ss.plot_shear_force(show=False, ax=ax_v)
                    st.pyplot(fig_v)
                with g3:
                    st.write("**Esforço Axial (N)**")
                    fig_n, ax_n = plt.subplots()
                    ss.plot_axial_force(show=False, ax=ax_n)
                    st.pyplot(fig_n)
                    
                st.success("🎯 Deslocamentos dos Nós Calculados:")
                for node in ss.get_node_displacements():
                    st.write(f"**Nó {node['id']}:** Ux = {node['ux']:.4f} m | Uy = {node['uy']:.4f} m | Rotação θz = {node['phi']:.4f} rad")

            except Exception as erro:
                st.error(f"Erro na montagem da matriz de rigidez: {erro}")
else:
    st.warning("Aguardando upload da foto para iniciar o mapeamento por visão computacional.")
